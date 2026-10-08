#include "cuda/cuda_jit.hpp"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace symft::cuda {
namespace {
std::string num(double value) {
    std::ostringstream s;
    s << std::setprecision(17) << std::scientific << value;
    return s.str();
}
std::string word(std::uint64_t v) { return std::to_string(v) + "ULL"; }
int insert_zero(int x, int bit) {
    int low = (1 << bit) - 1;
    return (x & low) | ((x & ~low) << 1);
}
void driver_check(CUresult r, const char* where) {
    if (r != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(r, &message);
        throw Error(std::string(where) + ": " + (message ? message : "CUDA driver error"));
    }
}

struct Generator {
    struct LinearBool {
        bool constant=false;
        std::set<int> terms;
        void add(const LinearBool& other) {
            constant^=other.constant;
            for(int t:other.terms) if(!terms.erase(t)) terms.insert(t);
        }
    };
    const CudaProgramData& p;
    std::ostringstream s;
    int k;
    int serial = 0;
    bool real_gauge = false;
    unsigned gauge = 0;
    std::vector<int> condition_map;
    int compact_conditions=0;
    bool symbolic=cuda_env_enabled("SYMFT_JIT_SYMBOLIC");
    int current_instruction=0;
    std::vector<std::map<int,LinearBool>> lowered;
    std::vector<LinearBool> detector_values;
    std::vector<std::vector<int>> symbolic_schedule;
    std::vector<int> branch_index;
    int branch_count=0;
    LinearBool logical_value;
    std::vector<LinearBool> initial_checks;
    std::set<int> live_branches;
    explicit Generator(const CudaProgramData& program) : p(program), k(p.initial_k), condition_map(p.symbol_words*64+1,-1) {
        // Exogenous noise symbols never live in this sampler. Pack only the
        // endogenous conditions actually read by residual expressions.
        for(const auto& m:p.residual_masks) {
            auto bits=m.mask;
            while(bits) {
                int bit=__builtin_ctzll(bits);int id=m.word*64+bit+1;
                if(condition_map.at(id)<0) condition_map[id]=++compact_conditions;
                bits&=bits-1;
            }
        }
        if(symbolic) compile_symbolic();
    }

    // Expand classical record assignments exactly over GF(2). Fresh stochastic
    // branches are independent leaves; no state probability is approximated.
    void compile_symbolic() {
        std::vector<LinearBool> conditions(p.symbol_words*64+1),records(p.record_words*64+1);
        lowered.resize(p.instructions.size());
        detector_values.resize(p.instructions.size());
        symbolic_schedule.resize(p.instructions.size()+1);
        branch_index.assign(p.instructions.size(),-1);
        auto evaluate=[&](int id) {
            const auto& e=p.expressions.at(id);
            LinearBool result;result.constant=e.residual_constant;
            if(e.block_expression>=0) result.terms.insert(e.block_expression);
            for(int j=0;j<e.residual_count;++j) {
                const auto& mask=p.residual_masks.at(e.residual_offset+j);
                auto bits=mask.mask;
                while(bits) {
                    int bit=__builtin_ctzll(bits);
                    result.add(conditions.at(mask.word*64+bit+1));
                    bits&=bits-1;
                }
            }
            return result;
        };
        for(int ix=0;ix<static_cast<int>(p.instructions.size());++ix) {
            const auto& i=p.instructions[ix];
            if(i.kind==CudaInstructionKind::ActiveMeasurement || i.kind==CudaInstructionKind::IntroduceDormantBranch) {
                branch_index[ix]=branch_count++;
                if(i.branch_condition>0) conditions.at(i.branch_condition)={false,{p.block_expression_count+ix}};
            }
            if(i.kind==CudaInstructionKind::ActiveRotationRun) {
                for(int j=0;j<i.rotation_run_count;++j) {
                    int id=p.rotation_run_items.at(i.rotation_run_offset+j).expression;
                    lowered[ix][id]=evaluate(id);
                }
            } else if(i.kind==CudaInstructionKind::RecordDetector) {
                LinearBool value;
                if(i.record_list_count) for(int j=0;j<i.record_list_count;++j)
                    value.add(records.at(p.record_table.at(i.record_list_offset+j)));
                else value=evaluate(i.expression);
                detector_values[ix]=value;
                int ready=-1;
                for(int t:value.terms) if(t>=p.block_expression_count)
                    ready=std::max(ready,t-p.block_expression_count);
                symbolic_schedule[ready+1].push_back(ix);
            } else {
                auto value=evaluate(i.expression);
                lowered[ix][i.expression]=value;
                if(i.kind==CudaInstructionKind::ActiveMeasurement || i.kind==CudaInstructionKind::IntroduceDormantBranch || i.kind==CudaInstructionKind::RecordMeasurement) {
                    if(i.record>0) records.at(i.record)=value;
                    if(i.record_condition>0) conditions.at(i.record_condition)=value;
                }
            }
        }
        for(std::size_t g=0;g<p.logical_group_offsets.size();++g)
            for(int j=0;j<p.logical_group_sizes[g];++j)
                logical_value.add(records.at(p.record_table.at(p.logical_group_offsets[g]+j)));
        // Row operations preserve the simultaneous zero-test. Reduce the
        // always-ready detector rows before evaluating them at runtime.
        std::map<int,LinearBool> rows;
        for(int ix:symbolic_schedule[0]) {
            auto row=detector_values[ix];
            while(!row.terms.empty()) {
                int pivot=*row.terms.rbegin();
                auto found=rows.find(pivot);
                if(found==rows.end()) {rows[pivot]=row;break;}
                row.add(found->second);
            }
            if(row.terms.empty() && row.constant) initial_checks.push_back(row);
        }
        for(auto it=rows.begin();it!=rows.end();++it)
            for(auto later=std::next(it);later!=rows.end();++later)
                if(later->second.terms.count(it->first)) later->second.add(it->second);
        for(const auto& [pivot,row]:rows) initial_checks.push_back(row);
        auto use=[&](const LinearBool& v) {
            for(int t:v.terms) if(t>=p.block_expression_count) live_branches.insert(t-p.block_expression_count);
        };
        for(std::size_t ix=0;ix<p.instructions.size();++ix) {
            auto kind=p.instructions[ix].kind;
            if(kind==CudaInstructionKind::ActiveRotation || kind==CudaInstructionKind::ActiveRotationRun || kind==CudaInstructionKind::PromoteDormantRotation)
                for(const auto& [id,value]:lowered[ix]) use(value);
            if(kind==CudaInstructionKind::RecordDetector) use(detector_values[ix]);
        }
        use(logical_value);
    }
    std::string linear_expr(const LinearBool& value) const {
        std::string result=value.constant?"true":"false";
        std::map<std::string,std::uint64_t> masks;
        for(int t:value.terms) {
            if(t<p.block_expression_count) {
                if(cuda_env_enabled("SYMFT_CUDA_PACKED_NOISE")) masks["ex"+std::to_string(t/64)]^=UintMask(t%64);
                else result+=" ^ bool((ex["+std::to_string(t)+"ULL*sw+(shot>>6)]>>(shot&63))&1ULL)";
            } else {
                int bit=branch_index.at(t-p.block_expression_count);
                masks["b"+std::to_string(bit/64)]^=UintMask(bit%64);
            }
        }
        for(const auto& [name,mask]:masks)
            result+=" ^ bool(__popcll("+name+" & "+word(mask)+")&1)";
        return "("+result+")";
    }

    bool supports_real_gauge() const {
        unsigned mask=0;
        int width=p.initial_k;
        auto rotation_ok=[&](int id) {
            const auto& r=p.rotations.at(id);
            if(r.minus_even_re!=0 && r.minus_even_im!=0) return false;
            return r.minus_even_im==0 ? (__builtin_parityll(mask&r.xmask)==0)
                                     : (__builtin_parityll(mask&r.xmask)==1);
        };
        for(const auto& i:p.instructions) {
            if(i.kind==CudaInstructionKind::ActiveRotation && !rotation_ok(i.rotation)) return false;
            if(i.kind==CudaInstructionKind::ActiveRotationRun)
                for(int j=0;j<i.rotation_run_count;++j)
                    if(!rotation_ok(p.rotation_run_items.at(i.rotation_run_offset+j).rotation)) return false;
            if(i.kind==CudaInstructionKind::PromoteDormantRotation) mask|=1U<<width++;
            if(i.kind==CudaInstructionKind::ActiveMeasurement) {
                const auto& m=p.measurements.at(i.measurement);
                if(m.is_diagonal) {
                    if(mask&(1U<<m.pivot)) mask^=m.zmask;
                } else {
                    if(m.even_phase_re!=0 && m.even_phase_im!=0) return false;
                    if(__builtin_parityll(mask&m.xmask)!=(m.even_phase_im!=0)) return false;
                }
                mask=(mask&((1U<<m.pivot)-1))|((mask>>(m.pivot+1))<<m.pivot);
                --width;
            }
        }
        return true;
    }

    std::string expr(int id) {
        if(symbolic) return linear_expr(lowered.at(current_instruction).at(id));
        const auto& e = p.expressions.at(id);
        std::string v = e.residual_constant ? "true" : "false";
        if (e.block_expression >= 0) {
            if(cuda_env_enabled("SYMFT_CUDA_PACKED_NOISE")) {
                v+=" ^ bool(ex"+std::to_string(e.block_expression/64)+" & "+word(UintMask(e.block_expression%64))+")";
            } else v += " ^ bool((ex[" + std::to_string(e.block_expression) +
                 "ULL*sw+(shot>>6)]>>(shot&63))&1ULL)";
        }
        std::map<int,std::uint64_t> masks;
        for (int i=0; i<e.residual_count; ++i) {
            const auto& m = p.residual_masks.at(e.residual_offset+i);
            auto bits=m.mask;
            while(bits) {
                int bit=__builtin_ctzll(bits);
                int id=condition_map.at(m.word*64+bit+1)-1;
                masks[id/64]^=std::uint64_t{1}<<(id%64);
                bits&=bits-1;
            }
        }
        for(const auto& [index,mask]:masks)
            v += " ^ bool(__popcll(c"+std::to_string(index)+" & "+word(mask)+")&1)";
        return "("+v+")";
    }
    static std::uint64_t UintMask(int bit) {return std::uint64_t{1}<<bit;}
    std::vector<std::vector<int>> early_detectors() const {
        if(symbolic) return symbolic_schedule;
        std::vector<std::vector<int>> schedule(p.instructions.size()+1);
        std::vector<int> conditions(p.symbol_words*64+1,-1),records(p.record_words*64+1,-1);
        for(int ix=0;ix<static_cast<int>(p.instructions.size());++ix) {
            const auto& i=p.instructions[ix];
            if(i.kind==CudaInstructionKind::RecordDetector) {
                int ready=-1;
                if(i.record_list_count) {
                    for(int j=0;j<i.record_list_count;++j)
                        ready=std::max(ready,records.at(p.record_table.at(i.record_list_offset+j)));
                } else {
                    const auto& e=p.expressions.at(i.expression);
                    for(int j=0;j<e.residual_count;++j) {
                        const auto& m=p.residual_masks.at(e.residual_offset+j);
                        auto bits=m.mask;
                        while(bits) {
                            int bit=__builtin_ctzll(bits);
                            ready=std::max(ready,conditions.at(m.word*64+bit+1));bits&=bits-1;
                        }
                    }
                }
                schedule[ready+1].push_back(ix);
            }
            if(i.kind==CudaInstructionKind::ActiveMeasurement || i.kind==CudaInstructionKind::IntroduceDormantBranch)
                if(i.branch_condition>0) conditions.at(i.branch_condition)=ix;
            if(i.kind==CudaInstructionKind::ActiveMeasurement || i.kind==CudaInstructionKind::IntroduceDormantBranch || i.kind==CudaInstructionKind::RecordMeasurement) {
                if(i.record>0)records.at(i.record)=ix;
                if(i.record_condition>0)conditions.at(i.record_condition)=ix;
            }
        }
        return schedule;
    }
    void setbit(const std::string& prefix, int bit, const std::string& value) {
        if(symbolic) {
            if(prefix!="c" || bit!=p.instructions.at(current_instruction).branch_condition) return;
            if(!live_branches.count(current_instruction)) return;
            int index=branch_index.at(current_instruction);
            if(index<0) return;
            auto mask=std::uint64_t{1}<<(index%64);
            s<<"b"<<index/64<<"=(b"<<index/64<<" & ~"<<word(mask)<<") | ("<<value<<" ? "<<word(mask)<<":0ULL);\n";
            return;
        }
        if (bit <= 0) return;
        if(prefix=="c") {bit=condition_map.at(bit);if(bit<=0)return;}
        std::string name = prefix + std::to_string((bit-1)/64);
        auto mask = std::uint64_t{1} << ((bit-1)%64);
        s << name << "=(" << name << " & ~" << word(mask) << ") | ("
          << value << " ? " << word(mask) << ":0ULL);\n";
    }
    void record(const CudaInstruction& i) {
        if(symbolic) return;
        s << "{bool outcome=" << expr(i.expression) << ";\n";
        setbit("m",i.record,"outcome");
        setbit("c",i.record_condition,"outcome");
        s << "}\n";
    }
    std::string parity_records(int offset, int count) {
        std::string v = "false";
        for (int j=0;j<count;++j) {
            int bit=p.record_table.at(offset+j)-1;
            v += " ^ bool((m"+std::to_string(bit/64)+">>"+std::to_string(bit%64)+")&1ULL)";
        }
        return "("+v+")";
    }
    void rotate(int id, int expression) {
        const auto& r=p.rotations.at(id);
        const int dim=1<<k, n=std::max(1,dim/32);
        s << "{bool sign=" << expr(expression) << ";\n";
        if(real_gauge) {
            unsigned mask=static_cast<unsigned>(r.zmask)^(r.minus_even_im!=0?gauge:0);
            double coeff=r.minus_even_im!=0?r.minus_even_im:r.minus_even_re;
            s<<"bool negative=sign";
            if(mask&31) s<<"^bool(__popc(lane&"<<(mask&31)<<")&1)";
            if(__builtin_parityll(r.xmask&mask)) s<<"^true";
            s<<";R q=negative?R("<<num(-coeff)<<"):R("<<num(coeff)<<");\n";
            for(int j=0;j<n;++j) {
                int other=j^(static_cast<int>(r.xmask)>>5);
                s<<"{R other=";
                if(r.xmask&31) s<<"__shfl_xor_sync(0xffffffffU,r"<<other<<","<<(r.xmask&31)<<")";
                else s<<"r"<<other;
                s<<";nr"<<j<<"=R("<<num(r.cos_angle)<<")*r"<<j
                 <<(__builtin_parity((j*32)&mask)?"-q*other":"+q*other")<<";}\n";
            }
            for(int j=0;j<n;++j) s<<"r"<<j<<"=nr"<<j<<";\n";
            s<<"}\n";
            return;
        }
        for(int j=0;j<n;++j) {
            int other=j^(static_cast<int>(r.xmask)>>5);
            s << "{R ar=r"<<j<<", ai=i"<<j<<";\n";
            if (r.xmask == 0) {
                s << "R br=ar,bi=ai;\n";
            } else if ((r.xmask&31)==0) {
                s << "R br=r"<<other<<",bi=i"<<other<<";\n";
            } else {
                s << "R br=__shfl_xor_sync(0xffffffffU,r"<<other<<","<<(r.xmask&31)
                  <<"),bi=__shfl_xor_sync(0xffffffffU,i"<<other<<","<<(r.xmask&31)<<");\n";
            }
            // Coefficient is evaluated on the SOURCE of the Pauli action.
            s << "bool odd=(__popc(((lane+"<<j*32<<")^"<<r.xmask<<") & "<<r.zmask<<")&1);\n"
              << "R direction=(sign!=odd)?R(-1):R(1);\n"
              << "R qr=direction*R("<<num(r.minus_even_re)<<"),qi=direction*R("<<num(r.minus_even_im)<<");\n"
              << "nr"<<j<<"=R("<<num(r.cos_angle)<<")*ar+qr*br-qi*bi;\n"
              << "ni"<<j<<"=R("<<num(r.cos_angle)<<")*ai+qr*bi+qi*br;\n}\n";
        }
        for(int j=0;j<n;++j) s<<"r"<<j<<"=nr"<<j<<";i"<<j<<"=ni"<<j<<";\n";
        s << "}\n";
    }
    void promote(const CudaInstruction& ins) {
        int dim=1<<k;
        s << "{R q="<<expr(ins.expression)<<"?R("<<num(ins.kernel_sin_angle)
          <<"):R("<<num(-ins.kernel_sin_angle)<<");\n";
        if(real_gauge) {
            if(k<5) {
                s<<"R a=__shfl_sync(0xffffffffU,r0,lane&"<<dim-1<<");\n"
                 <<"R signed_q=(__popc(lane&"<<gauge<<")&1)?-q:q;\n"
                 <<"r0=lane<"<<dim<<"?R("<<num(ins.kernel_cos_angle)<<")*a:signed_q*a;\n";
            } else {
                for(int j=0;j<dim/32;++j)
                    s<<"r"<<j+dim/32<<"=((__popc((lane+"<<j*32<<")&"<<gauge<<")&1)?-q:q)*r"<<j<<";r"<<j<<"*=R("<<num(ins.kernel_cos_angle)<<");\n";
            }
            s<<"}\n";
            gauge|=1U<<k++;
            return;
        }
        if(k<5) {
            s << "R a=__shfl_sync(0xffffffffU,r0,lane&"<<dim-1<<"),b=__shfl_sync(0xffffffffU,i0,lane&"<<dim-1<<");\n"
              << "r0=lane<"<<dim<<"?R("<<num(ins.kernel_cos_angle)<<")*a:-q*b;\n"
              << "i0=lane<"<<dim<<"?R("<<num(ins.kernel_cos_angle)<<")*b:q*a;\n";
        } else {
            int n=dim/32;
            for(int j=0;j<n;++j) s<<"r"<<j+n<<"=-q*i"<<j<<";i"<<j+n<<"=q*r"<<j<<";r"<<j<<"*=R("<<num(ins.kernel_cos_angle)<<");i"<<j<<"*=R("<<num(ins.kernel_cos_angle)<<");\n";
        }
        s<<"}\n";
        ++k;
    }
    // All shuffles are unconditional, including when the source register is
    // lane-dependent. This is required by the full-warp synchronization mask.
    std::pair<std::string,std::string> gather(const std::string& source,
                                             const std::set<int>& slots) {
        std::string tag="g"+std::to_string(serial++);
        s<<"int "<<tag<<"s="<<source<<";\n";
        for(int slot:slots) {
            s<<"R "<<tag<<"r"<<slot<<"=__shfl_sync(0xffffffffU,r"<<slot<<","<<tag<<"s&31);\n";
            if(!real_gauge) s<<"R "<<tag<<"i"<<slot<<"=__shfl_sync(0xffffffffU,i"<<slot<<","<<tag<<"s&31);\n";
        }
        std::string re="R(0)",im="R(0)";
        for(int slot:slots) {
            re="(("+tag+"s>>5)=="+std::to_string(slot)+"?"+tag+"r"+std::to_string(slot)+":"+re+")";
            im="(("+tag+"s>>5)=="+std::to_string(slot)+"?"+tag+"i"+std::to_string(slot)+":"+im+")";
        }
        return {re,im};
    }
    void measure(const CudaInstruction& ins) {
        const auto& m=p.measurements.at(ins.measurement);
        int n=std::max(1,m.out_dim/32);
        unsigned next_gauge=gauge;
        if(m.is_diagonal && (gauge&(1U<<m.pivot))) next_gauge^=m.zmask;
        next_gauge=(next_gauge&((1U<<m.pivot)-1))|((next_gauge>>(m.pivot+1))<<m.pivot);
        if(real_gauge && m.is_diagonal) {
            // Compute the probability in the input basis; gather only the
            // selected branch after the reduction. Neither candidate state
            // needs to remain live alongside the whole input vector.
            s<<"{R probability=0;\n";
            int dim=1<<k;
            for(int j=0;j<std::max(1,dim/32);++j) {
                s<<"if(";
                if(dim<32) s<<"lane<"<<dim<<" && ";
                s<<"(bool(__popc(lane&"<<(m.zmask&31)<<")&1)^"
                 <<(__builtin_parityll((j*32)&m.zmask)^m.diagonal_phase_bit?"true":"false")
                 <<")) probability+=r"<<j<<"*r"<<j<<";\n";
            }
            s<<"for(int delta=16;delta;delta>>=1) probability+=__shfl_down_sync(0xffffffffU,probability,delta);\n"
             <<"bool branch=false;R norm=0;if(lane==0){probability=fmin(R(1),fmax(R(0),probability));\n"
             <<"branch=sample(rng,probability);norm=branch?probability:R(1)-probability;norm=norm>R(0)?rsqrt(norm):R(0);}\n"
             <<"branch=__shfl_sync(0xffffffffU,int(branch),0);norm=__shfl_sync(0xffffffffU,norm,0);\n";
            setbit("c",ins.branch_condition,"branch");
            for(int j=0;j<n;++j) {
                s<<"{int packed=lane+"<<j*32<<";int base=(packed&"<<((1<<m.pivot)-1)<<")|((packed&~"<<((1<<m.pivot)-1)<<")<<1);\n";
                std::set<int> slots;
                for(int lane=0;lane<std::min(32,m.out_dim);++lane) {
                    int base=insert_zero(j*32+lane,m.pivot);
                    slots.insert(base>>5);slots.insert((base|(1<<m.pivot))>>5);
                }
                auto [re,im]=gather("base|(("+std::to_string(m.diagonal_phase_bit)+"^int(branch)^(__popc(base&"+std::to_string(m.zmask)+")&1))<<"+std::to_string(m.pivot)+")",slots);
                s<<"nr"<<j<<"="<<re<<"*norm;\n";
                if(gauge&(1U<<m.pivot))
                    s<<"if((branch^"<<(m.diagonal_phase_bit?"true":"false")<<") && (__popc(packed&"<<next_gauge<<")&1)) nr"<<j<<"=-nr"<<j<<";\n";
                s<<"}\n";
            }
            for(int j=0;j<n;++j) s<<"r"<<j<<"=nr"<<j<<";\n";
            s<<"}\n";
            --k;gauge=next_gauge;record(ins);
            return;
        }
        s<<"{R probability=0;\n";
        for(int j=0;j<n;++j) {
            s<<"R tr"<<j<<",ti"<<j<<",fr"<<j<<",fi"<<j<<";\n";
            s<<"{int packed=lane+"<<j*32<<";int base=(packed&"<<((1<<m.pivot)-1)<<")|((packed&~"<<((1<<m.pivot)-1)<<")<<1);\n";
            if(m.is_diagonal) {
                for(int branch=0;branch<2;++branch) {
                    std::set<int> slots;
                    for(int lane=0;lane<std::min(32,m.out_dim);++lane) {
                        int base=insert_zero(j*32+lane,m.pivot);
                        int b=m.diagonal_phase_bit ^ (__builtin_popcountll(base&m.zmask)&1) ^ branch;
                        slots.insert((base|(b<<m.pivot))>>5);
                    }
                    std::string source="base|(("+std::to_string(m.diagonal_phase_bit^branch)+"^(__popc(base&"+std::to_string(m.zmask)+")&1))<<"+std::to_string(m.pivot)+")";
                    auto [re,im]=gather(source,slots);
                    s<<(branch?"tr":"fr")<<j<<"=";
                    if(real_gauge && (gauge&(1U<<m.pivot)) && (m.diagonal_phase_bit^branch))
                        s<<"((__popc(packed&"<<next_gauge<<")&1)?R(-1):R(1))*";
                    s<<re<<";"<<(branch?"ti":"fi")<<j<<"="<<(real_gauge?"R(0)":im)<<";\n";
                }
            } else {
                std::set<int> slots0,slots1;
                for(int lane=0;lane<std::min(32,m.out_dim);++lane) {
                    int base=insert_zero(j*32+lane,m.pivot);
                    slots0.insert(base>>5);slots1.insert((base^m.xmask)>>5);
                }
                auto [r0,i0]=gather("base",slots0);
                auto [r1,i1]=gather("base^"+std::to_string(m.xmask),slots1);
                if(real_gauge) {
                    s<<"bool neg=bool(__popc(base&"<<m.zmask<<")&1)";
                    if(m.even_phase_im!=0) s<<"^bool(__popc((base^"<<m.xmask<<")&"<<gauge<<")&1)";
                    double coeff=m.even_phase_im!=0?-m.even_phase_im:m.even_phase_re;
                    s<<";R a=R(0.7071067811865475244)*"<<r0<<",b=(neg?R("<<num(-0.7071067811865475244*coeff)<<"):R("<<num(0.7071067811865475244*coeff)<<"))*"<<r1<<";\n"
                     <<"tr"<<j<<"=a-b;fr"<<j<<"=a+b;ti"<<j<<"=fi"<<j<<"=0;\n";
                } else s<<"R ar="<<r0<<",ai="<<i0<<",br="<<r1<<",bi="<<i1<<";\n"
                 <<"R sign=(__popc(base&"<<m.zmask<<")&1)?R(-1):R(1);\n"
                 <<"R qr=sign*R("<<num(0.7071067811865475244*m.even_phase_re)<<"),qi=sign*R("<<num(-0.7071067811865475244*m.even_phase_im)<<");\n"
                 <<"R a=R(0.7071067811865475244)*ar,b=R(0.7071067811865475244)*ai,c=qr*br-qi*bi,d=qr*bi+qi*br;\n"
                 <<"tr"<<j<<"=a-c;ti"<<j<<"=b-d;fr"<<j<<"=a+c;fi"<<j<<"=b+d;\n";
            }
            s<<"}\n";
            if(m.out_dim<32) s<<"if(lane<"<<m.out_dim<<") ";
            s<<"probability+=tr"<<j<<"*tr"<<j<<"+ti"<<j<<"*ti"<<j<<";\n";
        }
        s<<"for(int delta=16;delta;delta>>=1) probability+=__shfl_down_sync(0xffffffffU,probability,delta);\n"
          <<"bool branch=false;R norm=0;if(lane==0){probability=fmin(R(1),fmax(R(0),probability));\n"
          <<"branch=sample(rng,probability);norm=branch?probability:R(1)-probability;norm=norm>R(0)?rsqrt(norm):R(0);}\n"
          <<"branch=__shfl_sync(0xffffffffU,int(branch),0);norm=__shfl_sync(0xffffffffU,norm,0);\n";
        setbit("c",ins.branch_condition,"branch");
        for(int j=0;j<n;++j) {
            s<<"r"<<j<<"=(branch?tr"<<j<<":fr"<<j<<")*norm;";
            if(!real_gauge) s<<"i"<<j<<"=(branch?ti"<<j<<":fi"<<j<<")*norm;";
            s<<"\n";
        }
        s<<"}\n";
        --k;
        if(real_gauge) gauge=next_gauge;
        record(ins);
    }
    std::string generate() {
        real_gauge=supports_real_gauge() && !cuda_env_enabled("SYMFT_JIT_COMPLEX");
        s<<"// exact real gauge: "<<real_gauge<<"\n";
#ifdef SYMFT_CUDA_REAL_DOUBLE
        s<<"using R=double;\n";
#else
        s<<"using R=float;\n";
#endif
        s<<R"CUDA(
using U=unsigned long long;
__device__ __forceinline__ U next(U& state) {
    U z=(state+=0x9e3779b97f4a7c15ULL);
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;
    return z^(z>>31);
}
__device__ __forceinline__ bool sample(U& state,R p) {
    double uniform=double(next(state)>>11)*0x1.0p-53;
    if(p<=R(0)) return false;
    if(p>=R(1)) return true;
    return uniform<double(p);
}
extern "C" __global__ void symft_jit(const U* ex,U sw,int shots,U seed,int postselect,
                                     unsigned char* discarded,unsigned char* logical) {
    int lane=threadIdx.x&31;
    int shot=(blockIdx.x*blockDim.x+threadIdx.x)>>5;
    if(shot>=shots) return;
    U rng=seed^(0x9e3779b97f4a7c15ULL*U(shot+1));
    bool rejected=false;
)CUDA";
        if(cuda_env_enabled("SYMFT_CUDA_PACKED_NOISE")) {
            int words=(p.block_expression_count+63)/64;
            for(int w=0;w<words;++w) s<<"U ex"<<w<<"=ex[U(shot)*"<<words<<"+"<<w<<"];\n";
        }
        if(symbolic) {
            for(int j=0;j<(branch_count+63)/64;++j) s<<"U b"<<j<<"=0;\n";
        } else {
            for(int j=0;j<(compact_conditions+63)/64;++j) s<<"U c"<<j<<"=0;\n";
            for(int j=0;j<p.record_words;++j) s<<"U m"<<j<<"=0;\n";
        }
        for(int j=0;j<std::max(1,(1<<p.max_k)/32);++j)
            s<<"R r"<<j<<"="<<(j==0?"R(lane==0)":"R(0)")<<",i"<<j<<"=0,nr"<<j<<",ni"<<j<<";\n";
        auto schedule=early_detectors();
        for(std::size_t ix=0;ix<p.instructions.size();++ix) {
            current_instruction=static_cast<int>(ix);
            if(symbolic && ix==0 && cuda_env_enabled("SYMFT_JIT_ROW_REDUCE")) {
                s<<"// Equivalent row-reduced initial detector constraints\nif(postselect && (false";
                std::map<int,std::uint64_t> zero_masks;
                for(const auto& row:initial_checks) {
                    if(!row.constant && row.terms.size()==1 && cuda_env_enabled("SYMFT_CUDA_PACKED_NOISE")) {
                        int t=*row.terms.begin();zero_masks[t/64]|=UintMask(t%64);
                    } else s<<" || "<<linear_expr(row);
                }
                for(const auto& [word_id,mask]:zero_masks) s<<" || bool(ex"<<word_id<<" & "<<word(mask)<<")";
                s<<")) {if(lane==0){discarded[shot]=1;logical[shot]=0;}return;}\n";
            }
            for(int detector:schedule[ix]) {
                if(symbolic && ix==0 && cuda_env_enabled("SYMFT_JIT_ROW_REDUCE")) continue;
                const auto& d=p.instructions[detector];
                s<<"// earliest ready detector "<<detector<<"\nif(postselect && "
                 <<(symbolic?linear_expr(detector_values[detector]):(d.record_list_count?parity_records(d.record_list_offset,d.record_list_count):expr(d.expression)))
                 <<") {if(lane==0){discarded[shot]=1;logical[shot]=0;}return;}\n";
            }
            const auto& i=p.instructions[ix];
            s<<"// instruction "<<ix<<" kind "<<static_cast<int>(i.kind)<<" k="<<k<<"\n";
            switch(i.kind) {
            case CudaInstructionKind::ActiveRotation: rotate(i.rotation,i.expression);break;
            case CudaInstructionKind::ActiveRotationRun:
                for(int j=0;j<i.rotation_run_count;++j) {
                    const auto& r=p.rotation_run_items.at(i.rotation_run_offset+j);
                    rotate(r.rotation,r.expression);
                }
                break;
            case CudaInstructionKind::PromoteDormantRotation:promote(i);break;
            case CudaInstructionKind::ActiveMeasurement:measure(i);break;
            case CudaInstructionKind::RecordMeasurement:record(i);break;
            case CudaInstructionKind::IntroduceDormantBranch:
                if(symbolic && !live_branches.count(current_instruction)) {
                    s<<"if(lane==0) rng+=0x9e3779b97f4a7c15ULL; // unused draw, same counter advance\n";
                    break;
                }
                s<<"{int branch=0;if(lane==0)branch=next(rng)&1ULL;branch=__shfl_sync(0xffffffffU,branch,0);\n";setbit("c",i.branch_condition,"branch");s<<"}\n";record(i);break;
            case CudaInstructionKind::RecordDetector:
                s<<"if(!postselect && "<<(symbolic?linear_expr(detector_values[ix]):(i.record_list_count?parity_records(i.record_list_offset,i.record_list_count):expr(i.expression)))<<") rejected=true;\n";
                break;
            }
        }
        s<<"if(lane==0){discarded[shot]=rejected;logical[shot]=!rejected && (false";
        if(symbolic) s<<" ^ "<<linear_expr(logical_value);
        else for(std::size_t g=0;g<p.logical_group_offsets.size();++g)
                s<<" ^ "<<parity_records(p.logical_group_offsets[g],p.logical_group_sizes[g]);
        s<<");}\n}\n";
        return s.str();
    }
};
}

struct CudaJitSampler::Impl {
    bool packed=cuda_env_enabled("SYMFT_CUDA_PACKED_NOISE");
    CUmodule module=nullptr;
    CUfunction kernel=nullptr;
    ~Impl() { if(module) cuModuleUnload(module); }
};
CudaJitSampler::CudaJitSampler(const CudaProgramData& p):impl_(std::make_unique<Impl>()) {
    auto source=Generator(p).generate();
    if(const char* path=std::getenv("SYMFT_JIT_DUMP")) std::ofstream(path)<<source;
    int device=0;cudaGetDevice(&device);cudaDeviceProp prop{};cudaGetDeviceProperties(&prop,device);
    std::string arch="--gpu-architecture=sm_"+std::to_string(prop.major)+std::to_string(prop.minor);
    std::string regopt="--maxrregcount=";
    regopt+=std::getenv("SYMFT_JIT_REGS")?std::getenv("SYMFT_JIT_REGS"):"128";
    int nvmajor=0,nvminor=0;nvrtcVersion(&nvmajor,&nvminor);
    std::string identity=arch+regopt+std::to_string(nvmajor)+"."+std::to_string(nvminor)+"\n"+source;
    std::string cache;
    if(const char* root=std::getenv("SYMFT_JIT_CACHE")) {
        std::uint64_t hash=14695981039346656037ULL;
        for(unsigned char c:identity) {hash^=c;hash*=1099511628211ULL;}
        std::filesystem::create_directories(root);
        cache=(std::filesystem::path(root)/std::to_string(hash)).string();
        std::ifstream keyfile(cache+".key",std::ios::binary);
        std::string key((std::istreambuf_iterator<char>(keyfile)),{});
        if(key==identity) {
            std::ifstream binary(cache+".cubin",std::ios::binary);
            std::vector<char> code((std::istreambuf_iterator<char>(binary)),{});
            if(!code.empty() && cuModuleLoadData(&impl_->module,code.data())==CUDA_SUCCESS) {
                driver_check(cuModuleGetFunction(&impl_->kernel,impl_->module,"symft_jit"),"cached cuModuleGetFunction");
                return;
            }
        }
    }
    nvrtcProgram program=nullptr;
    auto created=nvrtcCreateProgram(&program,source.c_str(),"symft_jit.cu",0,nullptr,nullptr);
    if(created!=NVRTC_SUCCESS) throw Error(nvrtcGetErrorString(created));
    const char* args[]={"--std=c++17",arch.c_str(),"--device-as-default-execution-space",regopt.c_str()};
    auto status=nvrtcCompileProgram(program,regopt=="--maxrregcount=0"?3:4,args);
    std::size_t log_size=0;nvrtcGetProgramLogSize(program,&log_size);
    std::string log(log_size,'\0');nvrtcGetProgramLog(program,log.data());
    if(const char* path=std::getenv("SYMFT_JIT_DUMP")) std::ofstream(std::string(path)+".log")<<log;
    if(status!=NVRTC_SUCCESS) {nvrtcDestroyProgram(&program);throw Error("NVRTC: "+log);}
    std::size_t size=0;nvrtcGetCUBINSize(program,&size);std::vector<char> cubin(size);nvrtcGetCUBIN(program,cubin.data());
    nvrtcDestroyProgram(&program);
    if(!cache.empty()) {
        std::ofstream(cache+".cubin",std::ios::binary).write(cubin.data(),cubin.size());
        std::ofstream(cache+".key",std::ios::binary)<<identity;
    }
    driver_check(cuModuleLoadData(&impl_->module,cubin.data()),"cuModuleLoadData");
    driver_check(cuModuleGetFunction(&impl_->kernel,impl_->module,"symft_jit"),"cuModuleGetFunction");
    if(const char* path=std::getenv("SYMFT_JIT_DUMP")) {
        std::ofstream(std::string(path)+".cubin",std::ios::binary).write(cubin.data(),cubin.size());
        int regs=0,local=0;cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,impl_->kernel);cuFuncGetAttribute(&local,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,impl_->kernel);
        std::ofstream(std::string(path)+".resources")<<"registers "<<regs<<"\nlocal_bytes "<<local<<"\n";
    }
}
CudaJitSampler::~CudaJitSampler()=default;
bool CudaJitSampler::packed_expressions() const {return impl_->packed;}
void CudaJitSampler::launch(const std::uint64_t* expressions,std::size_t words,int shots,std::uint64_t seed,bool postselect,std::uint8_t* discarded,std::uint8_t* logical) const {
    int post=postselect;
    void* args[]={&expressions,&words,&shots,&seed,&post,&discarded,&logical};
    int threads=128;
    if(const char* value=std::getenv("SYMFT_JIT_THREADS")) threads=std::stoi(value);
    if(threads<32 || threads>1024 || threads%32) throw Error("SYMFT_JIT_THREADS must be a warp multiple in [32,1024]");
    driver_check(cuLaunchKernel(impl_->kernel,(shots+threads/32-1)/(threads/32),1,1,threads,1,1,0,nullptr,args,nullptr),"cuLaunchKernel symft_jit");
}
}
