#include "stabilizers.h"
#include <cassert>
#include <omp.h>
#include <random>
#include <tuple>
#include <algorithm>

const double PI = 3.14159265358979323846;

void Stabilizer::_x(int qubit){

    auto z_col = xt::col(this->tableau, this->num_qubits + qubit);
    auto phase = xt::col(this->tableau, this->num_qubits * 2);
    phase ^= z_col;
    return;
}

void Stabilizer::_y(int qubit){

    auto x_col = xt::col(this->tableau, qubit);
    auto z_col = xt::col(this->tableau, this->num_qubits + qubit);
    auto phase = xt::col(this->tableau, this->num_qubits * 2);
    phase ^= x_col ^ z_col;
    return;
}

void Stabilizer::_z(int qubit){

    auto x_col = xt::col(this->tableau, qubit);
    auto phase = xt::col(this->tableau, this->num_qubits * 2);
    phase ^= x_col;
    return;
}

void Stabilizer::_h(int qubit){

    auto x_col = xt::col(this->tableau, qubit);
    auto z_col = xt::col(this->tableau, this->num_qubits + qubit);
    auto phase = xt::col(this->tableau, this->num_qubits * 2);
    phase ^= (x_col & z_col);
    auto tmp = xt::eval(x_col);
    x_col = z_col;
    z_col = tmp;
    return;
}

void Stabilizer::_s(int qubit){

    auto x_col = xt::col(this->tableau, qubit);
    auto z_col = xt::col(this->tableau, this->num_qubits + qubit);
    auto phase = xt::col(this->tableau, this->num_qubits * 2);
    phase ^= (x_col & z_col);
    z_col ^= x_col;
    return;
}

void Stabilizer::_sdg(int qubit){

    auto x_col = xt::col(this->tableau, qubit);
    auto z_col = xt::col(this->tableau, this->num_qubits + qubit);
    auto phase = xt::col(this->tableau, this->num_qubits * 2);
    phase ^= (x_col & ~z_col);
    z_col ^= x_col;
    return;
}

void Stabilizer::_t(int qubit){

    std::vector<complex_t> coefs;
    std::vector<xt::xtensor<bool, 1>> destab_list;
    std::vector<xt::xtensor<bool, 1>> stab_list;
    this->tgate_decomp(coefs, destab_list, stab_list, qubit, false);
    /*
    std::cout << "coefs:\n";
    for (auto &k: coefs){
        std::cout << k << ' ';
    }
    std::cout << "\n";
    std::cout << "destab list:\n";
    for (auto &k: destab_list){
        std::cout << k << ' ';
    }
    std::cout << "\n";
    std::cout << "stab list:\n";
    for (auto &k: stab_list){
        std::cout << k << ' ';
    }
    std::cout << "\n";
    std::cout << "========" << std::endl;*/

    this->update_xvec(coefs, destab_list, stab_list);
    
}

void Stabilizer::_tdg(int qubit){

    std::vector<complex_t> coefs;
    std::vector<xt::xtensor<bool, 1>> destab_list;
    std::vector<xt::xtensor<bool, 1>> stab_list;
    this->tgate_decomp(coefs, destab_list, stab_list, qubit, true);
    /*
    std::cout << "coefs:\n";
    for (auto &k: coefs){
        std::cout << k << ' ';
    }
    std::cout << "\n";
    std::cout << "destab list:\n";
    for (auto &k: destab_list){
        std::cout << k << ' ';
    }
    std::cout << "\n";
    std::cout << "stab list:\n";
    for (auto &k: stab_list){
        std::cout << k << ' ';
    }
    std::cout << "\n";
    */
    this->update_xvec(coefs, destab_list, stab_list);
}

void Stabilizer::_cx(int c, int t){
    auto x0 = xt::col(this->tableau, c);
    auto z0 = xt::col(this->tableau, this->num_qubits + c);
    auto x1 = xt::col(this->tableau, t);
    auto z1 = xt::col(this->tableau, this->num_qubits + t);
    auto phase = xt::col(this->tableau, this->num_qubits * 2);

    phase ^= (x1 ^ z0 ^ true) & z1 & x0;
    x1 ^= x0;
    z0 ^= z1;
    return;
}


void Stabilizer::_reset(int qubit){

    auto res = this->_measure(qubit, true);
    if (res.reg == 1){
        this->_x(qubit);
    }
}

int Stabilizer::calc_g(bool x1, bool z1, bool x2, bool z2){
    
    if (!x1 && !z1){
        return 0;
    }else if (x1 && z1){
        return z2 - x2;
    }else if (x1 && !z1){
        return z2 * (2 * x2 - 1);
    }else{
        return x2 * (1 - 2 * z2);
    }
}


void Stabilizer::rowsum(int row1, int row2){
    auto h = xt::row(this->tableau, row1);
    auto i = xt::row(this->tableau, row2);
    int g_cnt = 0;
    int n = this->num_qubits;
    for (int j = 0; j < n; j++){
        g_cnt += calc_g(i(j), i(j+n), h(j), h(j+n));
    }
    bool res = ((2*h(2*n) + 2*i(2*n) + g_cnt) % 4 == 0);
    h ^= i;
    h(n * 2) = !res;
    return;
}

void Stabilizer::multiply_bool_pauli(complex_t&  phase, xt::xtensor<bool, 1>& pauli1, const xt::xtensor<bool, 1> pauli2){

    xt::xarray<complex_t> phase_mat = {
        { complex_t(1,0), complex_t(1,0),  complex_t(1,0),  complex_t(1,0) },
        { complex_t(1,0), complex_t(1,0),  complex_t(0,1),  complex_t(0,-1) },
        { complex_t(1,0), complex_t(0,-1), complex_t(1,0),  complex_t(0,1) },
        { complex_t(1,0), complex_t(0,1),  complex_t(0,-1), complex_t(1,0) }
    };

    for (int i = 0; i < this->num_qubits; i++){
        auto idx1 = pauli1(i) * 2 + pauli1(i+this->num_qubits);
        auto idx2 = pauli2(i) * 2 + pauli2(i+this->num_qubits);
        phase *= phase_mat(idx1, idx2);
    }
    phase *= (pauli2(2*this->num_qubits) ? complex_t(-1, 0) : complex_t(1,0));
    pauli1 ^= pauli2;
}

int Stabilizer::check_comm(xt::xtensor<bool, 1>& gate, xt::xtensor<bool, 1> entry, xt::xtensor<bool, 1> complement, complex_t& phase, xt::xtensor<bool, 1>& pauli, int qubit){

    int comm = 1;
    if (gate(qubit) == false && gate(qubit+this->num_qubits) == false) return 0;
    if (entry(qubit) == false && entry(qubit+this->num_qubits) == false) return 0;

    if (gate(qubit) != entry(qubit) || gate(qubit+this->num_qubits) != entry(qubit+this->num_qubits)){
        comm = -1;
    }

    if (comm > 0) return 0;

    this->multiply_bool_pauli(phase, pauli, complement);
    return 1;

}

void Stabilizer::gate_decomposition(xt::xtensor<bool, 1>& gate, xt::xtensor<bool, 1>& destab, xt::xtensor<bool, 1>& stab, complex_t& phase, int qubit){
    if (!xt::any(gate)) {
        return;
    }
    
    xt::xtensor<bool, 1> accum = xt::zeros<bool>({this->num_qubits * 2 + 1});
    for (int i = 0; i < this->num_qubits; i++){
        int res = this->check_comm(gate, xt::row(this->tableau, i), xt::row(this->tableau, i+this->num_qubits), phase, accum, qubit);
        stab(i) = bool(res);
    }
    for (int i = 0; i < this->num_qubits; i++){
        int res = this->check_comm(gate, xt::row(this->tableau, i+this->num_qubits), xt::row(this->tableau, i), phase, accum, qubit);
        destab(i) = bool(res);
    }

    return;
}

void Stabilizer::bin(long long value, xt::xtensor<bool, 1>& bin_v){

    int i = bin_v.size() - 1;
    while (value > 0){
        bin_v(i) = bool(value % 2);
        value /= 2;
        i -= 1;
    }
    return;
}

long long Stabilizer::convert(xt::xtensor<bool, 1>& bin_a, xt::xtensor<bool, 1>& bin_b){
    auto bin_t = bin_a ^ bin_b;
    int m = bin_a.size();
    long long res = 0;
    for (int i = 0; i < m; i++){
        res *= 2;
        if (bin_t(i)){
            res += 1;
        }
    }
    return res;
}

void Stabilizer::meas_tableau(xt::xtensor<bool, 1>& obs, xt::xtensor<bool, 1>& destab, xt::xtensor<bool, 1>& stab, int sign){
    int k = xt::from_indices(xt::argwhere(destab))(0, 0);
    
    for (int i = 0; i < this->num_qubits; i++){
        if ( i == k ) continue;
        if (destab(i)){
            this->rowsum(i+this->num_qubits, k+this->num_qubits);
        }
    }
    for ( int i = 0; i < this->num_qubits; i++){
        if (stab(i)){
            this->rowsum(i, k+this->num_qubits);
        }
        if (i == k){
            auto h = xt::row(this->tableau, i);
            h = xt::eval(xt::row(this->tableau, k+this->num_qubits));
        }
    }
    auto h = xt::row(this->tableau, k+this->num_qubits);
    h = xt::eval(obs);
    h(this->num_qubits * 2) = bool(sign); 
}

void Stabilizer::renorm(){

    double total = 0.;
    for (auto &kv : this->xvec) {
        total += (kv.second * std::conj(kv.second)).real();
    }
    total = std::sqrt(total);
    for (auto &kv : this->xvec) {
        kv.second /= total;
    }
}

MeasureResults Stabilizer::_measure(int qubit, bool reset_flag){

    xt::xtensor<bool, 1> obs = xt::zeros<bool>({this->num_qubits * 2 +1});
    obs(qubit+this->num_qubits) = true;
    xt::xtensor<bool, 1> destab = xt::zeros<bool>({this->num_qubits});
    xt::xtensor<bool, 1> stab = xt::zeros<bool>({this->num_qubits});

    complex_t phase = complex_t(1, 0);
    this->gate_decomposition(obs, destab, stab, phase, qubit);

    /*
    if (!reset_flag){
        std::cout << phase << std::endl;
        std::cout << destab << std::endl;
        std::cout << stab << std::endl;
    }*/

    std::unordered_map<long long, std::complex<double>> new_xvec_0;
    std::unordered_map<long long, std::complex<double>> new_xvec_1;

    std::vector<long long> keys;
    keys.reserve(this->xvec.size());
    for (auto &kv : this->xvec) { 
        keys.push_back(kv.first);
    }
    int keys_size = keys.size();
    
    std::mt19937 rng;                 // 默认使用 random_device 或固定种子均可
    rng.seed(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double out0 = 0.0;
    double out1 = 0.0;
    double ev   = 0.0;
    int outcome = 0;
    omp_threshold = this->omp_threshold;
    if (!xt::any(destab)){
        
        for (int i = 0; i < keys_size; i++) {
            long long c = keys[i];
            xt::xtensor<bool, 1> c_bin = xt::zeros<bool>({this->num_qubits});
            bin(c, c_bin);
            complex_t val = this->xvec[c];
            assert(std::abs(phase.imag()) < 1e-5);
            int tmp_v = ( xt::sum<int>(stab * c_bin)(0) % 2 == 0) ? 1 : -1;
            if (phase.real() * tmp_v > 0) 
            {
                new_xvec_0[c] += val;
                new_xvec_1[c] = 0;
                out0 += (val * std::conj(val)).real();
            } else {
                new_xvec_1[c] += val;
                new_xvec_0[c] = 0;
                out1 += (val * std::conj(val)).real();
            }
        }
        auto total = out0 + out1;
        if (std::abs(total - 1) > 1e6){
            out0 /= total;
            out1 /= total;
        }
        ev = out0 - out1;
        outcome = dist(rng) > out0 ? 1: 0;

    }else{
        
        xt::xtensor<bool, 1> k = xt::zeros<bool>({this->num_qubits});
        int idx = xt::from_indices(xt::argwhere(destab))(0, 0);
        k(idx) = true;
        
        /*
        #pragma omp parallel if( keys_size > omp_threshold)
        {
            std::unordered_map<long long, std::complex<double>> local_xvec_0;
                std::unordered_map<long long, std::complex<double>> local_xvec_1;
            double local_ev = 0.0;
            #pragma omp for nowait
            for (int i = 0; i < keys_size; i++) {
                long long c = keys[i];
                complex_t val = this->xvec[c];
                double coef = 1.0 / std::sqrt(2);
                complex_t coef0, coef1;
                xt::xtensor<bool, 1> c_bin = xt::zeros<bool>({this->num_qubits});
                bin(c, c_bin);
                double tmp_v = ( xt::sum<int>(stab * c_bin)(0) % 2 == 0) ? 1. : -1.;
                long long target_ind = c;
                if (xt::sum<int>(k * c_bin)(0) % 2 == 1){
                    coef0 = coef * phase * tmp_v;
                    coef1 = coef0 * -1.0;
                    target_ind = this->convert(c_bin, destab);
                }else{
                    coef0 = coef;
                    coef1 = coef;
                    target_ind = c;
                }
                local_xvec_0[target_ind] += val * coef0;
                local_xvec_1[target_ind] += val * coef1;
                std::cout << "update ev" << std::endl;
                if (this->xvec.find(target_ind) != this->xvec.end()){
                    std::cout << val << " " << std::conj(this->xvec[target_ind]) << std::endl;
                    auto v = phase * val * std::conj(this->xvec[target_ind]);
                    assert(abs(v.imag()) < 1e-5);
                    local_ev += v.real() * tmp_v;
                }
            } // end for
            // Now we do a critical section or parallel reduction:
            #pragma omp critical
            {
                ev += local_ev;
                for (auto &kv : local_xvec_0) {
                    new_xvec_0[kv.first] += kv.second;
                }
                for (auto &kv : local_xvec_1) {
                    new_xvec_1[kv.first] += kv.second;
                }
            }  
        } // end parallel
        */
        for (int i = 0; i < keys_size; i++) {
            long long c = keys[i];
            complex_t val = this->xvec[c];
            double coef = 1.0 / std::sqrt(2);
            complex_t coef0, coef1;
            xt::xtensor<bool, 1> c_bin = xt::zeros<bool>({this->num_qubits});
            bin(c, c_bin);
            double tmp_v = ( xt::sum<int>(stab * c_bin)(0) % 2 == 0) ? 1. : -1.;
            long long target_ind = c;
            if (xt::sum<int>(k * c_bin)(0) % 2 == 1){
                coef0 = coef * phase * tmp_v;
                coef1 = coef0 * -1.0;
                target_ind = this->convert(c_bin, destab);
            }else{
                coef0 = coef;
                coef1 = coef;
                target_ind = c;
            }
            new_xvec_0[target_ind] += val * coef0;
            new_xvec_1[target_ind] += val * coef1;
            auto t_ind = this->convert(c_bin, destab);
            if (this->xvec.find(t_ind) != this->xvec.end()){
                auto v = phase * val * std::conj(this->xvec[t_ind]);
                assert(abs(v.imag()) < 1e-5);
                ev += v.real() * tmp_v;
            }
        }
        out0 = (1+ev) / 2;
        out1 = (1-ev) / 2;
        outcome = dist(rng) > out0 ? 1: 0;

        this->meas_tableau(obs, destab, stab, outcome);
    }

    if (outcome){
        this->xvec = std::move(new_xvec_1);
    }else{
        this->xvec = std::move(new_xvec_0);
    }
    this->renorm();
    
    MeasureResults res;
    res.out0 = out0;
    res.out1 = out1;
    res.ev = out0 - out1;
    res.reg = outcome;
    /*
    if (!reset_flag){
        std::cout << "ev: " << res.ev << ", out0: " << res.out0 << ", out1: " << res.out1 << std::endl;
    }*/
    return res;
}

void Stabilizer::tgate_decomp(std::vector<complex_t>& coefs, std::vector<xt::xtensor<bool, 1>>& destab_list, std::vector<xt::xtensor<bool, 1>>& stab_list, int qubit, bool dag){

    coefs.push_back(complex_t(std::cos(PI/8)));
    destab_list.push_back(xt::zeros<bool>({this->num_qubits}));
    stab_list.push_back(xt::zeros<bool>({this->num_qubits}));
    
    double sign = dag ? -1.: 1.;
    
    xt::xtensor<bool, 1> gate = xt::zeros<bool>({this->num_qubits * 2 + 1});
    gate(qubit + this->num_qubits) = true;
    xt::xtensor<bool, 1> destab = xt::zeros<bool>({this->num_qubits});
    xt::xtensor<bool, 1> stab = xt::zeros<bool>({this->num_qubits});
    complex_t phase = complex_t(1, 0);
    this->gate_decomposition(gate, destab, stab, phase, qubit);
    coefs.push_back(complex_t(0, -1*std::sin(PI/8)) * phase * sign);
    destab_list.push_back(destab);
    stab_list.push_back(stab);
}

void Stabilizer::update_xvec(std::vector<complex_t>& coefs, std::vector<xt::xtensor<bool, 1>>& destab_list, std::vector<xt::xtensor<bool, 1>>& stab_list){
    
    int m = coefs.size();
    std::vector<long long> keys;
    keys.reserve(this->xvec.size());
    for (auto &kv : this->xvec) { 
        keys.push_back(kv.first);
    }
    int keys_size = keys.size();
    int omp_threshold = this->omp_threshold;
    std::unordered_map<long long, std::complex<double>> new_xvec;

    for (int i = 0; i < m; i++){
        auto co = coefs[i];
        if (std::abs(coefs[i]) < 1e-6) continue;
        /*
        #pragma omp parallel if( keys_size > omp_threshold)
        {
            std::unordered_map<long long, std::complex<double>> local_xvec;
            #pragma omp for nowait
            for (int j = 0; j < keys_size; j++) {
                long long c = keys[j];
                complex_t val = this->xvec[c];
                xt::xtensor<bool, 1> c_bin = xt::zeros<bool>({this->num_qubits});
                bin(c, c_bin);
                long long target_ind = this->convert(c_bin, destab_list[i]);
                int tmp_v = ( xt::sum<int>(stab_list[i] * c_bin)(0) % 2 == 0) ? 1 : -1;
                local_xvec[target_ind] += co * complex_t(tmp_v, 0) * val;
            } // end for
            // Now we do a critical section or parallel reduction:
            #pragma omp critical
            {
                for (auto &kv : local_xvec) {
                    new_xvec[kv.first] += kv.second;
                }
            }  
        } // end parallel
        */
        for (int j = 0; j < keys_size; j++) {
            long long c = keys[j];
            complex_t val = this->xvec[c];
            xt::xtensor<bool, 1> c_bin = xt::zeros<bool>({this->num_qubits});
            bin(c, c_bin);
            long long target_ind = this->convert(c_bin, destab_list[i]);
            auto tmp_v = ( xt::sum<int>(stab_list[i] * c_bin)(0) % 2 == 0) ? 1. : -1.;
            new_xvec[target_ind] += co * tmp_v * val;
            //std::cout << c << ' ' << target_ind << ' ' << new_xvec[target_ind] << std::endl;
        } // end for
    }

    this->xvec = std::move(new_xvec);
    this->renorm();
}


void Stabilizer::init(){
    this->tableau = xt::zeros<bool>({num_qubits * 2, num_qubits * 2 + 1});
    for(int i = 0; i < 2 * num_qubits; i++){
        this->tableau(i, i) = true;
    }
    this->xvec[0] = complex_t(1, 0);
}

std::vector<MeasureResults> Stabilizer::sim(std::vector<QuantumGate>& circuit){

    std::vector<MeasureResults> mres;

    for (const auto &op : circuit) {
        if ( op.gate == "X"){
            this->_x(op.targets[0]);
        }else if ( op.gate == "Y"){
            this->_y(op.targets[0]);
        }else if ( op.gate == "Z"){
            this->_z(op.targets[0]);
        }else if ( op.gate == "H"){
            this->_h(op.targets[0]);
        }else if ( op.gate == "S"){
            this->_s(op.targets[0]);
        }else if ( op.gate == "SDG"){
            this->_sdg(op.targets[0]);
        }else if ( op.gate == "T"){
            this->_t(op.targets[0]);
        }else if ( op.gate == "TDG"){
            this->_tdg(op.targets[0]);
        }else if ( op.gate == "R"){
            this->_reset(op.targets[0]);
        }else if (op.gate == "CX"){
            this->_cx(op.targets[0], op.targets[1]);
        }else if (op.gate == "M"){
            mres.push_back(this->_measure(op.targets[0], false));
        }else if (op.gate == "DETECTOR"){
            bool b = false;
            int n = mres.size();
            for (auto i: op.targets){
                if (mres[n+i].reg == 1){
                    b ^= true;
                }else{
                    b ^= false;
                }
            }
            if (b){
                std::cout << "detector error, discard the shot!";
                break;
            }
        }
        else{
            std::cout << op.gate << "is not support" << std::endl;
            assert(false);
        }
    }

    return mres;
}

void Stabilizer::print_xvec(){
    
    std::vector<std::tuple<long long, complex_t>> p;
    for (auto &k: this->xvec){
        if ((k.second * std::conj(k.second)).real() < 1e-5) continue;
        p.push_back({k.first, k.second});
    }

    // Define custom comparator
    auto tuple_comparator = [](const auto& a, const auto& b) {
        // First compare by the long long (first element)
        if (std::get<0>(a) != std::get<0>(b)) {
            return std::get<0>(a) < std::get<0>(b);
        }
        // If long longs are equal, compare by complex magnitude
        return std::abs(std::get<1>(a)) < std::abs(std::get<1>(b));
    };

    sort(p.begin(),p.end(), tuple_comparator);
    for(auto &i: p)
    {
        std::cout << std::get<0>(i) << ' ' << std::get<1>(i)  << std::endl;
    }
}

xt::xtensor<bool, 1> Stabilizer::phase(){
    return xt::col(this->tableau, this->num_qubits * 2);
}