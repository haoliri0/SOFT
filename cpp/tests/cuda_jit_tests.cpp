#include "cuda/cuda_jit.hpp"
#include "cuda/cuda_sampler.hpp"
#include "frontend/stim_prepared_sampler.hpp"
#include "sampler/exogenous.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using namespace symft;
using namespace symft::cuda;
using U=std::uint64_t;
using Z=std::complex<long double>;

void require(bool ok,const std::string& what) {if(!ok) throw Error(what);}
void check(cudaError_t r) {require(r==cudaSuccess,cudaGetErrorString(r));}
U next(U& s) {U z=(s+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
int insert(int value,int bit) {int low=(1<<bit)-1;return (value&low)|((value&~low)<<1);}
bool get(const std::vector<U>& w,int id) {return (w.at((id-1)/64)>>((id-1)%64))&1;}
void set(std::vector<U>& w,int id,bool v) {if(id>0){auto mask=U{1}<<((id-1)%64);auto& x=w.at((id-1)/64);x=(x&~mask)|(v?mask:0);}}

// Independent scalar long-double complex reference. It applies the Pauli
// matrix action to whole vectors rather than reproducing the warp/gauge code.
std::pair<int,int> reference(const CudaProgramData& p,const std::vector<U>& ex,
                             int sw,int shot,U seed,bool post) {
    std::vector<U> c(p.symbol_words),m(p.record_words);
    std::vector<Z> a(1<<p.initial_k);a[0]=1;
    U rng=seed^(0x9e3779b97f4a7c15ULL*U(shot+1));
    bool rejected=false;
    auto expression=[&](int id) {
        const auto& e=p.expressions.at(id);
        bool v=e.residual_constant;
        if(e.block_expression>=0) v^=(ex.at(e.block_expression*sw+shot/64)>>(shot%64))&1;
        for(int j=0;j<e.residual_count;++j) {
            const auto& x=p.residual_masks.at(e.residual_offset+j);
            v^=__builtin_parityll(c.at(x.word)&x.mask);
        }
        return v;
    };
    auto record=[&](const CudaInstruction& i) {bool b=expression(i.expression);set(m,i.record,b);set(c,i.record_condition,b);};
    auto rotate=[&](int id,int expression_id) {
        const auto& r=p.rotations.at(id);
        bool sign=expression(expression_id);
        std::vector<Z> b(a.size());
        for(std::size_t dst=0;dst<a.size();++dst) {
            auto src=dst^r.xmask;
            long double d=(sign^bool(__builtin_parityll(src&r.zmask)))?-1:1;
            b[dst]=static_cast<long double>(r.cos_angle)*a[dst]+d*Z(r.minus_even_re,r.minus_even_im)*a[src];
        }
        a=std::move(b);
    };
    for(const auto& i:p.instructions) {
        switch(i.kind) {
        case CudaInstructionKind::ActiveRotation:rotate(i.rotation,i.expression);break;
        case CudaInstructionKind::ActiveRotationRun:
            for(int j=0;j<i.rotation_run_count;++j) {auto r=p.rotation_run_items.at(i.rotation_run_offset+j);rotate(r.rotation,r.expression);}break;
        case CudaInstructionKind::PromoteDormantRotation: {
            std::vector<Z> b(a.size()*2);long double q=expression(i.expression)?i.kernel_sin_angle:-i.kernel_sin_angle;
            for(std::size_t j=0;j<a.size();++j){b[j]=static_cast<long double>(i.kernel_cos_angle)*a[j];b[j+a.size()]=Z(0,q)*a[j];}a=std::move(b);break;
        }
        case CudaInstructionKind::ActiveMeasurement: {
            const auto& q=p.measurements.at(i.measurement);
            auto project=[&](bool branch) {
                std::vector<Z> b(q.out_dim);
                for(int j=0;j<q.out_dim;++j) {
                    int base=insert(j,q.pivot);
                    if(q.is_diagonal) {
                        int pv=branch^q.diagonal_phase_bit^__builtin_parityll(base&q.zmask);
                        b[j]=a[base|(pv<<q.pivot)];
                    } else {
                        int sign=(branch^bool(__builtin_parityll(base&q.zmask)))?-1:1;
                        b[j]=(a[base]+static_cast<long double>(sign)*Z(q.even_phase_re,-q.even_phase_im)*a[base^q.xmask])/sqrtl(2);
                    }
                }
                return b;
            };
            auto b=project(true);long double prob=0;for(auto z:b)prob+=std::norm(z);
            prob=std::clamp(prob,0.L,1.L);
            long double u=static_cast<long double>(next(rng)>>11)*0x1.0p-53L;
            bool branch=u<prob;
            if(!branch)b=project(false);
            long double norm=branch?prob:1-prob;
            for(auto& z:b)z=norm>0?z/sqrtl(norm):Z(0);
            a=std::move(b);set(c,i.branch_condition,branch);record(i);break;
        }
        case CudaInstructionKind::IntroduceDormantBranch:set(c,i.branch_condition,next(rng)&1);record(i);break;
        case CudaInstructionKind::RecordMeasurement:record(i);break;
        case CudaInstructionKind::RecordDetector: {
            bool v=i.record_list_count?false:expression(i.expression);
            for(int j=0;j<i.record_list_count;++j)v^=get(m,p.record_table.at(i.record_list_offset+j));
            if(v){rejected=true;if(post)return {1,0};}break;
        }
        }
    }
    bool logical=false;
    for(std::size_t g=0;g<p.logical_group_offsets.size();++g)
        for(int j=0;j<p.logical_group_sizes[g];++j)logical^=get(m,p.record_table.at(p.logical_group_offsets[g]+j));
    return {rejected,!rejected&&logical};
}

CudaProgramData prepare(const CircuitSamplingInput& input) {
    PackedPresampledExogenous samples;
    PresampledExpressionPlan plan;
    prepare_presampled_exogenous_packed(samples,input.program);
    prepare_presampled_expression_plan(plan,input.program,samples);
    return build_cuda_program_data(input.program,plan,input.logical_records);
}

void compare(CudaProgramData p,const std::string& name,int shots,bool post) {
    int sw=(shots+63)/64;
    std::mt19937_64 random(1234567);
    std::vector<U> ex(p.block_expression_count*sw);for(auto& x:ex)x=random();
    std::vector<U> upload=ex;
    if(std::getenv("SYMFT_CUDA_PACKED_NOISE")) {
        int words=(p.block_expression_count+63)/64;
        upload.assign(shots*words,0);
        for(int shot=0;shot<shots;++shot) for(int e=0;e<p.block_expression_count;++e)
            if((ex[e*sw+shot/64]>>(shot%64))&1)upload[shot*words+e/64]|=U{1}<<(e%64);
    }
    U* dex=nullptr;unsigned char *dd=nullptr,*dl=nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&dex),std::max(std::size_t(8),upload.size()*sizeof(U))));
    if(!upload.empty())check(cudaMemcpy(dex,upload.data(),upload.size()*sizeof(U),cudaMemcpyHostToDevice));
    check(cudaMalloc(reinterpret_cast<void**>(&dd),shots));check(cudaMalloc(reinterpret_cast<void**>(&dl),shots));
    CudaJitSampler sampler(p);
    for(U seed:{17ULL,997ULL,0x1234567812345678ULL}) {
        sampler.launch(dex,sw,shots,seed,post,dd,dl);
        std::vector<unsigned char>d(shots),l(shots);
        check(cudaMemcpy(d.data(),dd,shots,cudaMemcpyDeviceToHost));check(cudaMemcpy(l.data(),dl,shots,cudaMemcpyDeviceToHost));
        for(int shot=0;shot<shots;++shot) {
            auto expected=reference(p,ex,sw,shot,seed,post);
            require(d[shot]==expected.first && l[shot]==expected.second,
                    name+" differs from long-double reference at shot "+std::to_string(shot)+" seed "+std::to_string(seed));
        }
    }
    cudaFree(dex);cudaFree(dd);cudaFree(dl);
    std::cout<<"PASS dense-reference "<<name<<" max_k="<<p.max_k<<" shots="<<shots*3<<"\n";
}

int main() {
    try {
        for(const auto& [name,text]:std::vector<std::pair<std::string,std::string>>{
            {"real_T","H 0\nT 0\nH 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n"},
            {"complex_TS","H 0\nT 0\nS 0\nH 0\nT 0\nH 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n"},
            {"feedback_Y","H 0 1\nT 0\nT_DAG 1\nCX 0 1\nMY 0\nCY rec[-1] 1\nH 1\nT 1\nM 1\nOBSERVABLE_INCLUDE(0) rec[-1]\n"},
            {"postselection","H 0 1\nT 0 1\nCX 0 1\nMX 0\nDETECTOR rec[-1]\nMY 1\nOBSERVABLE_INCLUDE(0) rec[-1]\n"}}) {
            auto input=make_stim_circuit_sampling_input(parse_stim_circuit_text(text));
            compare(prepare(input),name,257,true);
            if(name=="postselection") compare(prepare(input),name+"_disabled",257,false);
        }
        // More than one packed branch word; transitive feedback cancels in
        // repeated detector rows, while a late non-Clifford outcome remains.
        std::string algebra;
        for(int q=0;q<70;++q) algebra+="H "+std::to_string(q)+"\nM "+std::to_string(q)+"\n";
        algebra+="X_ERROR(0.1) 70\nCX rec[-1] 70 rec[-2] 70 rec[-3] 70\nM 70\n"
                 "DETECTOR rec[-1] rec[-2] rec[-3] rec[-4]\n"
                 "DETECTOR rec[-1] rec[-2] rec[-3] rec[-4]\n"
                 "H 71\nT 71\nCX rec[-1] 71\nH 71\nM 71\n"
                 "OBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-35] rec[-71]\n";
        auto algebra_plan=prepare(make_stim_circuit_sampling_input(parse_stim_circuit_text(algebra)));
        compare(algebra_plan,"gf2_feedback_70_branches",257,true);
        compare(algebra_plan,"gf2_feedback_70_branches_disabled",257,false);
        compare(prepare(make_stim_circuit_sampling_input(parse_stim_circuit_text(
            "X_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nX 0\nM 0\n"
            "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"))),
            "gf2_contradictory_rows",257,true);
        for(const auto& file:{"msc_d3_inject_cultivate_p1e-3.stim","msc_d5_inject_cultivate_p1e-3.stim"}) {
            auto input=make_stim_circuit_sampling_input_from_file(std::string("benchmark/circuit/")+file);
            auto p=prepare(input);
            // Remove postselection only in the validation copy so every random
            // expression pattern executes the whole state evolution. Expose
            // parity of all records as the validation observable.
            p.instructions.erase(std::remove_if(p.instructions.begin(),p.instructions.end(),[](const auto& i){return i.kind==CudaInstructionKind::RecordDetector;}),p.instructions.end());
            p.logical_group_offsets={static_cast<int>(p.record_table.size())};
            p.logical_group_sizes={input.program.nrecords};
            for(int i=1;i<=input.program.nrecords;++i)p.record_table.push_back(i);
            compare(p,std::string(file)+"_all_records",129,false);
            if(p.max_k==10 && !std::getenv("SYMFT_JIT_COMPLEX")) {
                setenv("SYMFT_JIT_COMPLEX","1",1);
                compare(p,std::string(file)+"_forced_complex",65,false);
                unsetenv("SYMFT_JIT_COMPLEX");
            }
        }
        std::cout<<"cuda_jit_tests passed\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
