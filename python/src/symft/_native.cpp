#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <numpy/arrayobject.h>

#include "core/common.hpp"
#include "frontend/stim_prepared_sampler.hpp"
#include "sampler/active.hpp"
#include "sampler/batch_sampler.hpp"
#include "sampler/prepared_sampler.hpp"
#include "sampler/single_shot.hpp"

#ifdef SYMFT_CPP_ENABLE_CUDA
#include "cuda/cuda_runtime.hpp"
#include "cuda/cuda_sampler.hpp"
#endif

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

PyObject* SymFTError = nullptr;

struct AllowThreads {
    PyThreadState* state;

    AllowThreads() : state(PyEval_SaveThread()) {}
    ~AllowThreads() { PyEval_RestoreThread(state); }

    AllowThreads(const AllowThreads&) = delete;
    AllowThreads& operator=(const AllowThreads&) = delete;
};

struct AllowThreadsWithLock {
    PyThreadState* state;
    PyThread_type_lock lock;

    explicit AllowThreadsWithLock(PyThread_type_lock lock_)
        : state(PyEval_SaveThread()), lock(lock_) {
        PyThread_acquire_lock(lock, WAIT_LOCK);
    }
    ~AllowThreadsWithLock() {
        PyThread_release_lock(lock);
        PyEval_RestoreThread(state);
    }

    AllowThreadsWithLock(const AllowThreadsWithLock&) = delete;
    AllowThreadsWithLock& operator=(const AllowThreadsWithLock&) = delete;
};

struct PyCircuit {
    PyObject_HEAD
    symft::QuantumCircuit* circuit;
};

struct PyCompiledMeasurementSampler {
    PyObject_HEAD
    symft::FactoredInstructionProgram* program;
    int use_batch;
    int batch_size;
    int sample_chunk_shots;
};

struct PyCompiledCountsSampler {
    PyObject_HEAD
    symft::PreparedCircuitSingleShotSampler* single;
    symft::PreparedCircuitBatchSampler* batch;
#ifdef SYMFT_CPP_ENABLE_CUDA
    symft::cuda::PreparedCircuitCudaSampler* cuda;
#endif
    int use_batch;
    int use_cuda;
    PyThread_type_lock lock;
};

struct PyCudaSamplingOptions {
    bool postselect_detectors = false;
    bool sample_exogenous_on_device = true;
    bool gpu_presample_expressions = false;
    bool lazy_exogenous_on_device = false;
    bool gpu_on_demand_expressions = false;
    int shots_per_launch = 0;
    int threads_per_block = 0;
};

PyTypeObject CircuitType = {PyVarObject_HEAD_INIT(nullptr, 0)};
PyTypeObject CompiledMeasurementSamplerType = {PyVarObject_HEAD_INIT(nullptr, 0)};
PyTypeObject CompiledCountsSamplerType = {PyVarObject_HEAD_INIT(nullptr, 0)};

int set_cpp_exception() {
    try {
        throw;
    } catch (const symft::Error& ex) {
        PyErr_SetString(SymFTError, ex.what());
    } catch (const std::exception& ex) {
        PyErr_SetString(PyExc_RuntimeError, ex.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unknown SymFT C++ exception");
    }
    return -1;
}

PyObject* set_cpp_exception_null() {
    set_cpp_exception();
    return nullptr;
}

bool object_to_string(PyObject* object, std::string& out, const char* name) {
    if (PyUnicode_Check(object)) {
        Py_ssize_t size = 0;
        const char* data = PyUnicode_AsUTF8AndSize(object, &size);
        if (data == nullptr) {
            return false;
        }
        out.assign(data, static_cast<std::size_t>(size));
        return true;
    }
    if (PyBytes_Check(object)) {
        char* data = nullptr;
        Py_ssize_t size = 0;
        if (PyBytes_AsStringAndSize(object, &data, &size) < 0) {
            return false;
        }
        out.assign(data, static_cast<std::size_t>(size));
        return true;
    }
    PyErr_Format(PyExc_TypeError, "%s must be str or bytes", name);
    return false;
}

bool path_to_string(PyObject* object, std::string& out) {
    PyObject* path = PyOS_FSPath(object);
    if (path == nullptr) {
        return false;
    }
    const bool ok = object_to_string(path, out, "path");
    Py_DECREF(path);
    return ok;
}

bool dict_set_owned(PyObject* dict, const char* key, PyObject* value) {
    if (value == nullptr) {
        return false;
    }
    const int ok = PyDict_SetItemString(dict, key, value);
    Py_DECREF(value);
    return ok == 0;
}

bool parse_nonnegative_int_arg(long long value, int& out, const char* name) {
    if (value < 0) {
        PyErr_Format(PyExc_ValueError, "%s must be nonnegative", name);
        return false;
    }
    if (value > static_cast<long long>(std::numeric_limits<int>::max())) {
        PyErr_Format(PyExc_OverflowError, "%s exceeds the supported C++ int range", name);
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

symft::CircuitSamplingOptions make_options(
    int observable,
    int postselect_detectors,
    int sample_chunk_shots,
    int batch_size,
    int batch_mask_threshold_denominator,
    int threads) {
    symft::CircuitSamplingOptions options;
    options.observable = observable;
    options.postselect_detectors = postselect_detectors != 0;
    options.sample_chunk_shots = sample_chunk_shots;
    options.batch_size = batch_size;
    options.batch_mask_threshold_denominator = batch_mask_threshold_denominator;
    options.threads = threads;
    return options;
}

std::string normalized_cuda_mode(std::string mode) {
    std::string out;
    out.reserve(mode.size());
    for (unsigned char ch : mode) {
        out.push_back(ch == '-' ? '_' : static_cast<char>(std::tolower(ch)));
    }
    return out;
}

void clear_cuda_exogenous_modes(PyCudaSamplingOptions& options) {
    options.sample_exogenous_on_device = false;
    options.gpu_presample_expressions = false;
    options.lazy_exogenous_on_device = false;
    options.gpu_on_demand_expressions = false;
}

bool configure_cuda_mode(PyCudaSamplingOptions& options, const std::string& raw_mode) {
    const std::string mode = normalized_cuda_mode(raw_mode);
    clear_cuda_exogenous_modes(options);
    if (mode == "gpu" || mode == "gpu_exogenous" || mode == "device" || mode == "device_exogenous") {
        options.sample_exogenous_on_device = true;
        return true;
    }
    if (mode == "cpu" || mode == "cpu_presampled" || mode == "cpu_presample_exogenous") {
        return true;
    }
    if (mode == "gpu_presample_expressions" || mode == "gpu_presampled_expressions") {
        options.gpu_presample_expressions = true;
        return true;
    }
    if (mode == "gpu_on_demand" || mode == "gpu_on_demand_expressions" || mode == "on_demand") {
        options.gpu_on_demand_expressions = true;
        return true;
    }
    if (mode == "gpu_lazy" || mode == "gpu_lazy_exogenous" || mode == "lazy" || mode == "lazy_exogenous") {
        options.lazy_exogenous_on_device = true;
        return true;
    }
    PyErr_Format(
        PyExc_ValueError,
        "cuda_mode must be one of 'gpu', 'cpu_presampled', "
        "'gpu_presample_expressions', 'gpu_on_demand_expressions', or 'gpu_lazy', not '%s'",
        raw_mode.c_str());
    return false;
}

bool make_py_cuda_options(
    int postselect_detectors,
    int shots_per_launch,
    int threads_per_block,
    PyObject* mode_object,
    PyCudaSamplingOptions& options) {
    options.postselect_detectors = postselect_detectors != 0;
    options.shots_per_launch = shots_per_launch;
    options.threads_per_block = threads_per_block;

    std::string mode = "gpu";
    if (mode_object != nullptr && mode_object != Py_None) {
        if (!object_to_string(mode_object, mode, "cuda_mode")) {
            return false;
        }
    }
    return configure_cuda_mode(options, mode);
}

#ifdef SYMFT_CPP_ENABLE_CUDA
symft::cuda::CudaSamplingOptions cpp_cuda_options_from_py(const PyCudaSamplingOptions& py_options) {
    symft::cuda::CudaSamplingOptions options;
    options.postselect_detectors = py_options.postselect_detectors;
    options.sample_exogenous_on_device = py_options.sample_exogenous_on_device;
    options.gpu_presample_expressions = py_options.gpu_presample_expressions;
    options.lazy_exogenous_on_device = py_options.lazy_exogenous_on_device;
    options.gpu_on_demand_expressions = py_options.gpu_on_demand_expressions;
    options.shots_per_launch = py_options.shots_per_launch;
    options.threads_per_block = py_options.threads_per_block;
    return options;
}
#endif

symft::FactoredInstructionProgram make_program_from_circuit(const symft::QuantumCircuit& circuit) {
    symft::CircuitSamplingOptions options;
    symft::CircuitSamplingInput input = symft::make_stim_circuit_sampling_input(circuit, options);
    return std::move(input.program);
}

PyObject* words_to_numpy(
    const std::vector<std::vector<std::uint64_t>>& rows,
    int nbits,
    bool bit_packed) {
    if (nbits < 0) {
        PyErr_SetString(PyExc_ValueError, "number of bits must be nonnegative");
        return nullptr;
    }
    const npy_intp nshots = static_cast<npy_intp>(rows.size());
    const npy_intp width = bit_packed
                               ? static_cast<npy_intp>((nbits + 7) >> 3)
                               : static_cast<npy_intp>(nbits);
    npy_intp dims[2] = {nshots, width};
    PyObject* array = PyArray_SimpleNew(2, dims, bit_packed ? NPY_UINT8 : NPY_BOOL);
    if (array == nullptr) {
        return nullptr;
    }

    auto* data = static_cast<unsigned char*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)));
    const std::size_t total = static_cast<std::size_t>(nshots) * static_cast<std::size_t>(width);
    std::fill(data, data + total, 0);

    if (width == 0) {
        return array;
    }

    for (std::size_t shot = 0; shot < rows.size(); ++shot) {
        const auto& row = rows[shot];
        if (bit_packed) {
            unsigned char* out = data + shot * static_cast<std::size_t>(width);
            for (int bit = 0; bit < nbits; ++bit) {
                if (symft::packed_bit(row, bit)) {
                    out[static_cast<std::size_t>(bit >> 3)] |=
                        static_cast<unsigned char>(1u << (bit & 7));
                }
            }
        } else {
            unsigned char* out = data + shot * static_cast<std::size_t>(width);
            for (int bit = 0; bit < nbits; ++bit) {
                out[static_cast<std::size_t>(bit)] =
                    static_cast<unsigned char>(symft::packed_bit(row, bit) ? 1 : 0);
            }
        }
    }
    return array;
}

std::vector<std::vector<std::uint64_t>> sample_detector_words_single_shot(
    const symft::FactoredInstructionProgram& program,
    int shots,
    std::uint64_t seed) {
    symft::FactoredExecutorState runtime(program, seed);
    std::vector<std::vector<std::uint64_t>> out;
    out.reserve(static_cast<std::size_t>(shots));
    for (int shot = 0; shot < shots; ++shot) {
        (void)shot;
        symft::reset_executor(runtime, program);
        symft::execute_in_place(runtime, program);
        out.push_back(runtime.detector_words);
    }
    return out;
}

std::vector<std::vector<std::uint64_t>> sample_measurement_words(
    const symft::FactoredInstructionProgram& program,
    int shots,
    std::uint64_t seed,
    int use_batch,
    int batch_size,
    int sample_chunk_shots) {
    if (use_batch) {
        return symft::sample_measurements_batch(program, shots, batch_size, seed);
    }
    return symft::sample_measurements(program, shots, seed, sample_chunk_shots);
}

PyObject* timing_to_dict(const symft::CircuitSamplingTiming& timing) {
    PyObject* dict = PyDict_New();
    if (dict == nullptr) {
        return nullptr;
    }
    if (!dict_set_owned(dict, "parse_s", PyFloat_FromDouble(timing.parse_s)) ||
        !dict_set_owned(dict, "plan_s", PyFloat_FromDouble(timing.plan_s)) ||
        !dict_set_owned(dict, "presample_s", PyFloat_FromDouble(timing.presample_s)) ||
        !dict_set_owned(dict, "execute_s", PyFloat_FromDouble(timing.execute_s)) ||
        !dict_set_owned(dict, "accumulate_s", PyFloat_FromDouble(timing.accumulate_s)) ||
        !dict_set_owned(dict, "sample_s", PyFloat_FromDouble(timing.sample_s))) {
        Py_DECREF(dict);
        return nullptr;
    }
    return dict;
}

double safe_ratio(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

PyObject* counts_result_to_dict(const symft::CircuitSamplingRunResult& result) {
    PyObject* dict = PyDict_New();
    if (dict == nullptr) {
        return nullptr;
    }

    const auto& counts = result.counts;
    if (!dict_set_owned(dict, "shots", PyLong_FromUnsignedLongLong(counts.shots)) ||
        !dict_set_owned(dict, "discarded", PyLong_FromUnsignedLongLong(counts.discarded)) ||
        !dict_set_owned(dict, "accepted", PyLong_FromUnsignedLongLong(counts.accepted)) ||
        !dict_set_owned(dict, "logical_errors", PyLong_FromUnsignedLongLong(counts.logical_errors)) ||
        !dict_set_owned(dict, "discard_rate", PyFloat_FromDouble(safe_ratio(counts.discarded, counts.shots))) ||
        !dict_set_owned(
            dict,
            "logical_error_rate",
            PyFloat_FromDouble(safe_ratio(counts.logical_errors, counts.accepted))) ||
        !dict_set_owned(dict, "active_threads", PyLong_FromLong(result.active_threads))) {
        Py_DECREF(dict);
        return nullptr;
    }

    PyObject* timing = timing_to_dict(result.timing);
    if (timing == nullptr) {
        Py_DECREF(dict);
        return nullptr;
    }
    PyDict_SetItemString(dict, "timing", timing);
    Py_DECREF(timing);
    return dict;
}

PyObject* info_to_dict(const symft::CircuitSamplingInfo& info) {
    PyObject* dict = PyDict_New();
    if (dict == nullptr) {
        return nullptr;
    }
    if (!dict_set_owned(dict, "num_qubits", PyLong_FromLong(info.n)) ||
        !dict_set_owned(dict, "num_measurements", PyLong_FromLong(info.records)) ||
        !dict_set_owned(dict, "num_detectors", PyLong_FromLong(info.detectors)) ||
        !dict_set_owned(dict, "num_observable_includes", PyLong_FromLong(info.observable_includes)) ||
        !dict_set_owned(dict, "observable", PyLong_FromLong(info.observable)) ||
        !dict_set_owned(dict, "max_active_qubits", PyLong_FromLong(info.max_k)) ||
        !dict_set_owned(dict, "batch_size", PyLong_FromLong(info.batch_size)) ||
        !dict_set_owned(dict, "sample_chunk_shots", PyLong_FromLong(info.sample_chunk_shots)) ||
        !dict_set_owned(dict, "threads", PyLong_FromLong(info.threads)) ||
        !dict_set_owned(dict, "active_components", PyBool_FromLong(info.active_components)) ||
        !dict_set_owned(dict, "detector_postselection", PyBool_FromLong(info.detector_postselection)) ||
        !dict_set_owned(
            dict,
            "batch_mask_threshold_denominator",
            PyLong_FromLong(info.batch_mask_threshold_denominator))) {
        Py_DECREF(dict);
        return nullptr;
    }
    return dict;
}

int num_observables(const symft::QuantumCircuit& circuit) {
    int out = 0;
    for (const auto& observable : circuit.observables) {
        out = std::max(out, observable.index + 1);
    }
    return out;
}

PyObject* create_measurement_sampler(
    symft::FactoredInstructionProgram&& program,
    int use_batch,
    int batch_size,
    int sample_chunk_shots) {
    auto* self = reinterpret_cast<PyCompiledMeasurementSampler*>(
        CompiledMeasurementSamplerType.tp_alloc(&CompiledMeasurementSamplerType, 0));
    if (self == nullptr) {
        return nullptr;
    }
    self->program = nullptr;
    try {
        self->program = new symft::FactoredInstructionProgram(std::move(program));
    } catch (...) {
        Py_DECREF(reinterpret_cast<PyObject*>(self));
        return set_cpp_exception_null();
    }
    self->use_batch = use_batch;
    self->batch_size = batch_size;
    self->sample_chunk_shots = sample_chunk_shots;
    return reinterpret_cast<PyObject*>(self);
}

PyObject* create_counts_sampler(
    const symft::QuantumCircuit& circuit,
    int use_batch,
    int use_cuda,
    const symft::CircuitSamplingOptions& options,
    const PyCudaSamplingOptions& cuda_options) {
#ifndef SYMFT_CPP_ENABLE_CUDA
    if (use_cuda) {
        PyErr_SetString(
            SymFTError,
            "CUDA support was not compiled into symft._native; rebuild with SYMFT_PY_ENABLE_CUDA=1");
        return nullptr;
    }
#endif

    auto* self = reinterpret_cast<PyCompiledCountsSampler*>(
        CompiledCountsSamplerType.tp_alloc(&CompiledCountsSamplerType, 0));
    if (self == nullptr) {
        return nullptr;
    }
    self->single = nullptr;
    self->batch = nullptr;
#ifdef SYMFT_CPP_ENABLE_CUDA
    self->cuda = nullptr;
#endif
    self->use_batch = use_batch;
    self->use_cuda = use_cuda;
    self->lock = PyThread_allocate_lock();
    if (self->lock == nullptr) {
        Py_DECREF(reinterpret_cast<PyObject*>(self));
        return PyErr_NoMemory();
    }

    try {
        AllowThreads allow;
        symft::CircuitSamplingInput input = symft::make_stim_circuit_sampling_input(circuit, options);
        if (use_cuda) {
#ifdef SYMFT_CPP_ENABLE_CUDA
            self->cuda = new symft::cuda::PreparedCircuitCudaSampler(
                std::move(input),
                cpp_cuda_options_from_py(cuda_options));
#else
            (void)input;
#endif
        } else if (use_batch) {
            self->batch = new symft::PreparedCircuitBatchSampler(std::move(input), options);
        } else {
            self->single = new symft::PreparedCircuitSingleShotSampler(std::move(input), options);
        }
    } catch (...) {
        Py_DECREF(reinterpret_cast<PyObject*>(self));
        return set_cpp_exception_null();
    }
    return reinterpret_cast<PyObject*>(self);
}

void Circuit_dealloc(PyCircuit* self) {
    delete self->circuit;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

int Circuit_init(PyCircuit* self, PyObject* args, PyObject* kwargs) {
    PyObject* text_object = nullptr;
    PyObject* path_object = nullptr;
    static const char* kwlist[] = {"text", "path", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|OO:Circuit",
            const_cast<char**>(kwlist),
            &text_object,
            &path_object)) {
        return -1;
    }

    if (text_object != nullptr && path_object != nullptr && path_object != Py_None) {
        PyErr_SetString(PyExc_TypeError, "Circuit accepts text or path, not both");
        return -1;
    }

    if (self->circuit != nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "Circuit objects cannot be reinitialized");
        return -1;
    }

    try {
        symft::QuantumCircuit parsed;
        if (path_object != nullptr && path_object != Py_None) {
            std::string path;
            if (!path_to_string(path_object, path)) {
                return -1;
            }
            parsed = symft::parse_stim_circuit_file(path);
        } else {
            std::string text;
            if (text_object != nullptr && text_object != Py_None) {
                if (!object_to_string(text_object, text, "text")) {
                    return -1;
                }
            }
            parsed = symft::parse_stim_circuit_text(text);
        }
        auto circuit = std::make_unique<symft::QuantumCircuit>(std::move(parsed));
        self->circuit = circuit.release();
    } catch (...) {
        return set_cpp_exception();
    }
    return 0;
}

PyObject* Circuit_repr(PyCircuit* self) {
    const auto& circuit = *self->circuit;
    return PyUnicode_FromFormat(
        "<symft.Circuit qubits=%d measurements=%d detectors=%zd observables=%d>",
        circuit.nqubits,
        circuit.nrecords,
        circuit.detectors.size(),
        num_observables(circuit));
}

PyObject* Circuit_compile_sampler(PyCircuit* self, PyObject* args, PyObject* kwargs) {
    int use_batch = 0;
    long long batch_size_value = 0;
    long long sample_chunk_value = 0;
    static const char* kwlist[] = {"batch", "batch_size", "sample_chunk_shots", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|pLL:compile_sampler",
            const_cast<char**>(kwlist),
            &use_batch,
            &batch_size_value,
            &sample_chunk_value)) {
        return nullptr;
    }

    int batch_size = 0;
    int sample_chunk_shots = 0;
    if (!parse_nonnegative_int_arg(batch_size_value, batch_size, "batch_size") ||
        !parse_nonnegative_int_arg(sample_chunk_value, sample_chunk_shots, "sample_chunk_shots")) {
        return nullptr;
    }

    try {
        symft::FactoredInstructionProgram program;
        {
            AllowThreads allow;
            program = make_program_from_circuit(*self->circuit);
        }
        return create_measurement_sampler(std::move(program), use_batch, batch_size, sample_chunk_shots);
    } catch (...) {
        return set_cpp_exception_null();
    }
}

PyObject* Circuit_compile_counts_sampler(PyCircuit* self, PyObject* args, PyObject* kwargs) {
    int use_batch = 1;
    long long observable_value = 0;
    int postselect_detectors = 0;
    long long batch_size_value = 0;
    long long sample_chunk_value = 0;
    long long threads_value = 1;
    long long threshold_value = 2;
    int use_cuda = 0;
    PyObject* cuda_mode_object = Py_None;
    long long shots_per_launch_value = 0;
    long long threads_per_block_value = 0;
    static const char* kwlist[] = {
        "batch",
        "observable",
        "postselect_detectors",
        "batch_size",
        "sample_chunk_shots",
        "threads",
        "batch_mask_threshold_denominator",
        "cuda",
        "cuda_mode",
        "shots_per_launch",
        "threads_per_block",
        nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|pLpLLLLpOLL:compile_counts_sampler",
            const_cast<char**>(kwlist),
            &use_batch,
            &observable_value,
            &postselect_detectors,
            &batch_size_value,
            &sample_chunk_value,
            &threads_value,
            &threshold_value,
            &use_cuda,
            &cuda_mode_object,
            &shots_per_launch_value,
            &threads_per_block_value)) {
        return nullptr;
    }

    int observable = 0;
    int batch_size = 0;
    int sample_chunk_shots = 0;
    int threads = 1;
    int threshold = 2;
    int shots_per_launch = 0;
    int threads_per_block = 0;
    if (!parse_nonnegative_int_arg(observable_value, observable, "observable") ||
        !parse_nonnegative_int_arg(batch_size_value, batch_size, "batch_size") ||
        !parse_nonnegative_int_arg(sample_chunk_value, sample_chunk_shots, "sample_chunk_shots") ||
        !parse_nonnegative_int_arg(threads_value, threads, "threads") ||
        !parse_nonnegative_int_arg(threshold_value, threshold, "batch_mask_threshold_denominator") ||
        !parse_nonnegative_int_arg(shots_per_launch_value, shots_per_launch, "shots_per_launch") ||
        !parse_nonnegative_int_arg(threads_per_block_value, threads_per_block, "threads_per_block")) {
        return nullptr;
    }

    PyCudaSamplingOptions cuda_options;
    if (!make_py_cuda_options(
            postselect_detectors,
            shots_per_launch,
            threads_per_block,
            cuda_mode_object,
            cuda_options)) {
        return nullptr;
    }

    const auto options = make_options(
        observable,
        postselect_detectors,
        sample_chunk_shots,
        batch_size,
        threshold,
        threads);
    return create_counts_sampler(*self->circuit, use_batch, use_cuda, options, cuda_options);
}

PyObject* Circuit_sample(PyCircuit* self, PyObject* args, PyObject* kwargs) {
    long long shots_value = 1;
    unsigned long long seed = 1;
    int use_batch = 0;
    int bit_packed = 0;
    long long batch_size_value = 0;
    long long sample_chunk_value = 0;
    static const char* kwlist[] = {
        "shots",
        "seed",
        "batch",
        "bit_packed",
        "batch_size",
        "sample_chunk_shots",
        nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|LKppLL:sample",
            const_cast<char**>(kwlist),
            &shots_value,
            &seed,
            &use_batch,
            &bit_packed,
            &batch_size_value,
            &sample_chunk_value)) {
        return nullptr;
    }

    int shots = 0;
    int batch_size = 0;
    int sample_chunk_shots = 0;
    if (!parse_nonnegative_int_arg(shots_value, shots, "shots") ||
        !parse_nonnegative_int_arg(batch_size_value, batch_size, "batch_size") ||
        !parse_nonnegative_int_arg(sample_chunk_value, sample_chunk_shots, "sample_chunk_shots")) {
        return nullptr;
    }

    int nrecords = 0;
    std::vector<std::vector<std::uint64_t>> rows;
    try {
        AllowThreads allow;
        symft::FactoredInstructionProgram program = make_program_from_circuit(*self->circuit);
        nrecords = program.nrecords;
        rows = sample_measurement_words(
            program,
            shots,
            static_cast<std::uint64_t>(seed),
            use_batch,
            batch_size,
            sample_chunk_shots);
    } catch (...) {
        return set_cpp_exception_null();
    }
    return words_to_numpy(rows, nrecords, bit_packed != 0);
}

PyObject* Circuit_sample_counts(PyCircuit* self, PyObject* args, PyObject* kwargs) {
    long long shots_value = 1;
    unsigned long long seed = 1;
    int use_batch = 1;
    long long observable_value = 0;
    int postselect_detectors = 0;
    long long batch_size_value = 0;
    long long sample_chunk_value = 0;
    long long threads_value = 1;
    long long threshold_value = 2;
    int use_cuda = 0;
    PyObject* cuda_mode_object = Py_None;
    long long shots_per_launch_value = 0;
    long long threads_per_block_value = 0;
    static const char* kwlist[] = {
        "shots",
        "seed",
        "batch",
        "observable",
        "postselect_detectors",
        "batch_size",
        "sample_chunk_shots",
        "threads",
        "batch_mask_threshold_denominator",
        "cuda",
        "cuda_mode",
        "shots_per_launch",
        "threads_per_block",
        nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|LKpLpLLLLpOLL:sample_counts",
            const_cast<char**>(kwlist),
            &shots_value,
            &seed,
            &use_batch,
            &observable_value,
            &postselect_detectors,
            &batch_size_value,
            &sample_chunk_value,
            &threads_value,
            &threshold_value,
            &use_cuda,
            &cuda_mode_object,
            &shots_per_launch_value,
            &threads_per_block_value)) {
        return nullptr;
    }

    int shots = 0;
    int observable = 0;
    int batch_size = 0;
    int sample_chunk_shots = 0;
    int threads = 1;
    int threshold = 2;
    int shots_per_launch = 0;
    int threads_per_block = 0;
    if (!parse_nonnegative_int_arg(shots_value, shots, "shots") ||
        !parse_nonnegative_int_arg(observable_value, observable, "observable") ||
        !parse_nonnegative_int_arg(batch_size_value, batch_size, "batch_size") ||
        !parse_nonnegative_int_arg(sample_chunk_value, sample_chunk_shots, "sample_chunk_shots") ||
        !parse_nonnegative_int_arg(threads_value, threads, "threads") ||
        !parse_nonnegative_int_arg(threshold_value, threshold, "batch_mask_threshold_denominator") ||
        !parse_nonnegative_int_arg(shots_per_launch_value, shots_per_launch, "shots_per_launch") ||
        !parse_nonnegative_int_arg(threads_per_block_value, threads_per_block, "threads_per_block")) {
        return nullptr;
    }

    PyCudaSamplingOptions cuda_options;
    if (!make_py_cuda_options(
            postselect_detectors,
            shots_per_launch,
            threads_per_block,
            cuda_mode_object,
            cuda_options)) {
        return nullptr;
    }

    const auto options = make_options(
        observable,
        postselect_detectors,
        sample_chunk_shots,
        batch_size,
        threshold,
        threads);
    PyObject* sampler_object = create_counts_sampler(*self->circuit, use_batch, use_cuda, options, cuda_options);
    if (sampler_object == nullptr) {
        return nullptr;
    }
    auto* sampler = reinterpret_cast<PyCompiledCountsSampler*>(sampler_object);
    symft::CircuitSamplingRunResult result;
    try {
        AllowThreads allow;
        if (sampler->use_cuda) {
#ifdef SYMFT_CPP_ENABLE_CUDA
            result = sampler->cuda->sample(static_cast<std::uint64_t>(shots), seed);
#endif
        } else if (sampler->use_batch) {
            result = sampler->batch->sample(static_cast<std::uint64_t>(shots), seed);
        } else {
            result = sampler->single->sample(static_cast<std::uint64_t>(shots), seed);
        }
    } catch (...) {
        Py_DECREF(sampler_object);
        return set_cpp_exception_null();
    }
    Py_DECREF(sampler_object);
    return counts_result_to_dict(result);
}

PyObject* Circuit_sample_detectors(PyCircuit* self, PyObject* args, PyObject* kwargs) {
    long long shots_value = 1;
    unsigned long long seed = 1;
    int bit_packed = 0;
    static const char* kwlist[] = {"shots", "seed", "bit_packed", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|LKp:sample_detectors",
            const_cast<char**>(kwlist),
            &shots_value,
            &seed,
            &bit_packed)) {
        return nullptr;
    }

    int shots = 0;
    if (!parse_nonnegative_int_arg(shots_value, shots, "shots")) {
        return nullptr;
    }

    int ndetectors = 0;
    std::vector<std::vector<std::uint64_t>> rows;
    try {
        AllowThreads allow;
        symft::FactoredInstructionProgram program = make_program_from_circuit(*self->circuit);
        ndetectors = program.ndetectors;
        rows = sample_detector_words_single_shot(
            program,
            shots,
            static_cast<std::uint64_t>(seed));
    } catch (...) {
        return set_cpp_exception_null();
    }
    return words_to_numpy(rows, ndetectors, bit_packed != 0);
}

PyObject* Circuit_get_num_qubits(PyCircuit* self, void*) {
    return PyLong_FromLong(self->circuit->nqubits);
}

PyObject* Circuit_get_num_measurements(PyCircuit* self, void*) {
    return PyLong_FromLong(self->circuit->nrecords);
}

PyObject* Circuit_get_num_detectors(PyCircuit* self, void*) {
    return PyLong_FromSsize_t(static_cast<Py_ssize_t>(self->circuit->detectors.size()));
}

PyObject* Circuit_get_num_observables(PyCircuit* self, void*) {
    return PyLong_FromLong(num_observables(*self->circuit));
}

PyObject* Circuit_get_num_observable_includes(PyCircuit* self, void*) {
    return PyLong_FromSsize_t(static_cast<Py_ssize_t>(self->circuit->observables.size()));
}

PyObject* Circuit_get_detectors(PyCircuit* self, void*) {
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(self->circuit->detectors.size()));
    if (list == nullptr) {
        return nullptr;
    }
    for (std::size_t i = 0; i < self->circuit->detectors.size(); ++i) {
        const auto& detector = self->circuit->detectors[i];
        PyObject* dict = PyDict_New();
        PyObject* records = PyTuple_New(static_cast<Py_ssize_t>(detector.records.size()));
        PyObject* coords = PyTuple_New(static_cast<Py_ssize_t>(detector.coords.size()));
        if (dict == nullptr || records == nullptr || coords == nullptr) {
            Py_XDECREF(dict);
            Py_XDECREF(records);
            Py_XDECREF(coords);
            Py_DECREF(list);
            return nullptr;
        }
        for (std::size_t j = 0; j < detector.records.size(); ++j) {
            PyTuple_SET_ITEM(records, static_cast<Py_ssize_t>(j), PyLong_FromLong(detector.records[j]));
        }
        for (std::size_t j = 0; j < detector.coords.size(); ++j) {
            PyTuple_SET_ITEM(coords, static_cast<Py_ssize_t>(j), PyFloat_FromDouble(detector.coords[j]));
        }
        if (PyDict_SetItemString(dict, "records", records) < 0 ||
            PyDict_SetItemString(dict, "coords", coords) < 0 ||
            !dict_set_owned(dict, "line", PyLong_FromLong(detector.line))) {
            Py_DECREF(records);
            Py_DECREF(coords);
            Py_DECREF(dict);
            Py_DECREF(list);
            return nullptr;
        }
        Py_DECREF(records);
        Py_DECREF(coords);
        PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), dict);
    }
    return list;
}

PyObject* Circuit_get_observables(PyCircuit* self, void*) {
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(self->circuit->observables.size()));
    if (list == nullptr) {
        return nullptr;
    }
    for (std::size_t i = 0; i < self->circuit->observables.size(); ++i) {
        const auto& observable = self->circuit->observables[i];
        PyObject* dict = PyDict_New();
        PyObject* records = PyTuple_New(static_cast<Py_ssize_t>(observable.records.size()));
        if (dict == nullptr || records == nullptr) {
            Py_XDECREF(dict);
            Py_XDECREF(records);
            Py_DECREF(list);
            return nullptr;
        }
        for (std::size_t j = 0; j < observable.records.size(); ++j) {
            PyTuple_SET_ITEM(records, static_cast<Py_ssize_t>(j), PyLong_FromLong(observable.records[j]));
        }
        if (!dict_set_owned(dict, "index", PyLong_FromLong(observable.index)) ||
            PyDict_SetItemString(dict, "records", records) < 0 ||
            !dict_set_owned(dict, "line", PyLong_FromLong(observable.line))) {
            Py_DECREF(records);
            Py_DECREF(dict);
            Py_DECREF(list);
            return nullptr;
        }
        Py_DECREF(records);
        PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), dict);
    }
    return list;
}

void CompiledMeasurementSampler_dealloc(PyCompiledMeasurementSampler* self) {
    delete self->program;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* CompiledMeasurementSampler_repr(PyCompiledMeasurementSampler* self) {
    return PyUnicode_FromFormat(
        "<symft.CompiledMeasurementSampler qubits=%d measurements=%d detectors=%d max_active_qubits=%d batch=%s>",
        self->program->n,
        self->program->nrecords,
        self->program->ndetectors,
        self->program->max_k,
        self->use_batch ? "True" : "False");
}

PyObject* CompiledMeasurementSampler_sample(PyCompiledMeasurementSampler* self, PyObject* args, PyObject* kwargs) {
    long long shots_value = 1;
    unsigned long long seed = 1;
    int bit_packed = 0;
    static const char* kwlist[] = {"shots", "seed", "bit_packed", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|LKp:sample",
            const_cast<char**>(kwlist),
            &shots_value,
            &seed,
            &bit_packed)) {
        return nullptr;
    }

    int shots = 0;
    if (!parse_nonnegative_int_arg(shots_value, shots, "shots")) {
        return nullptr;
    }

    std::vector<std::vector<std::uint64_t>> rows;
    try {
        AllowThreads allow;
        if (self->use_batch) {
            rows = symft::sample_measurements_batch(
                *self->program,
                shots,
                self->batch_size,
                static_cast<std::uint64_t>(seed));
        } else {
            rows = symft::sample_measurements(
                *self->program,
                shots,
                static_cast<std::uint64_t>(seed),
                self->sample_chunk_shots);
        }
    } catch (...) {
        return set_cpp_exception_null();
    }
    return words_to_numpy(rows, self->program->nrecords, bit_packed != 0);
}

PyObject* CompiledMeasurementSampler_sample_detectors(
    PyCompiledMeasurementSampler* self,
    PyObject* args,
    PyObject* kwargs) {
    long long shots_value = 1;
    unsigned long long seed = 1;
    int bit_packed = 0;
    static const char* kwlist[] = {"shots", "seed", "bit_packed", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|LKp:sample_detectors",
            const_cast<char**>(kwlist),
            &shots_value,
            &seed,
            &bit_packed)) {
        return nullptr;
    }

    int shots = 0;
    if (!parse_nonnegative_int_arg(shots_value, shots, "shots")) {
        return nullptr;
    }

    std::vector<std::vector<std::uint64_t>> rows;
    try {
        AllowThreads allow;
        rows = sample_detector_words_single_shot(
            *self->program,
            shots,
            static_cast<std::uint64_t>(seed));
    } catch (...) {
        return set_cpp_exception_null();
    }
    return words_to_numpy(rows, self->program->ndetectors, bit_packed != 0);
}

PyObject* CompiledMeasurementSampler_get_num_qubits(PyCompiledMeasurementSampler* self, void*) {
    return PyLong_FromLong(self->program->n);
}

PyObject* CompiledMeasurementSampler_get_num_measurements(PyCompiledMeasurementSampler* self, void*) {
    return PyLong_FromLong(self->program->nrecords);
}

PyObject* CompiledMeasurementSampler_get_num_detectors(PyCompiledMeasurementSampler* self, void*) {
    return PyLong_FromLong(self->program->ndetectors);
}

PyObject* CompiledMeasurementSampler_get_max_active_qubits(PyCompiledMeasurementSampler* self, void*) {
    return PyLong_FromLong(self->program->max_k);
}

void CompiledCountsSampler_dealloc(PyCompiledCountsSampler* self) {
    delete self->single;
    delete self->batch;
#ifdef SYMFT_CPP_ENABLE_CUDA
    delete self->cuda;
#endif
    if (self->lock != nullptr) {
        PyThread_free_lock(self->lock);
    }
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

const symft::CircuitSamplingInfo& CompiledCountsSampler_info_ref(PyCompiledCountsSampler* self) {
#ifdef SYMFT_CPP_ENABLE_CUDA
    if (self->use_cuda) {
        return self->cuda->info();
    }
#endif
    return self->use_batch ? self->batch->info() : self->single->info();
}

const symft::CircuitSamplingTiming& CompiledCountsSampler_preprocessing_timing_ref(
    PyCompiledCountsSampler* self) {
#ifdef SYMFT_CPP_ENABLE_CUDA
    if (self->use_cuda) {
        return self->cuda->preprocessing_timing();
    }
#endif
    return self->use_batch
               ? self->batch->preprocessing_timing()
               : self->single->preprocessing_timing();
}

const char* CompiledCountsSampler_backend_name(PyCompiledCountsSampler* self) {
    if (self->use_cuda) {
        return "cuda";
    }
    return self->use_batch ? "batch" : "single";
}

PyObject* CompiledCountsSampler_repr(PyCompiledCountsSampler* self) {
    const auto& info = CompiledCountsSampler_info_ref(self);
    return PyUnicode_FromFormat(
        "<symft.CompiledCountsSampler qubits=%d measurements=%d detectors=%d max_active_qubits=%d backend=%s>",
        info.n,
        info.records,
        info.detectors,
        info.max_k,
        CompiledCountsSampler_backend_name(self));
}

PyObject* CompiledCountsSampler_sample(PyCompiledCountsSampler* self, PyObject* args, PyObject* kwargs) {
    long long shots_value = 1;
    PyObject* stream_id_object = Py_None;
    static const char* kwlist[] = {"shots", "stream_id", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|LO:sample",
            const_cast<char**>(kwlist),
            &shots_value,
            &stream_id_object)) {
        return nullptr;
    }

    int shots = 0;
    if (!parse_nonnegative_int_arg(shots_value, shots, "shots")) {
        return nullptr;
    }

    const bool has_stream_id = stream_id_object != nullptr && stream_id_object != Py_None;
    unsigned long long stream_id = 0;
    if (has_stream_id) {
        stream_id = PyLong_AsUnsignedLongLong(stream_id_object);
        if (PyErr_Occurred()) {
            return nullptr;
        }
    }

    symft::CircuitSamplingRunResult result;
    try {
        AllowThreadsWithLock allow(self->lock);
        if (self->use_cuda) {
#ifdef SYMFT_CPP_ENABLE_CUDA
            result = has_stream_id
                         ? self->cuda->sample(static_cast<std::uint64_t>(shots), stream_id)
                         : self->cuda->sample(static_cast<std::uint64_t>(shots));
#endif
        } else if (self->use_batch) {
            result = has_stream_id
                         ? self->batch->sample(static_cast<std::uint64_t>(shots), stream_id)
                         : self->batch->sample(static_cast<std::uint64_t>(shots));
        } else {
            result = has_stream_id
                         ? self->single->sample(static_cast<std::uint64_t>(shots), stream_id)
                         : self->single->sample(static_cast<std::uint64_t>(shots));
        }
    } catch (...) {
        return set_cpp_exception_null();
    }
    return counts_result_to_dict(result);
}

PyObject* CompiledCountsSampler_get_info(PyCompiledCountsSampler* self, void*) {
    PyObject* dict = info_to_dict(CompiledCountsSampler_info_ref(self));
    if (dict == nullptr) {
        return nullptr;
    }
    if (!dict_set_owned(dict, "backend", PyUnicode_FromString(CompiledCountsSampler_backend_name(self)))) {
        Py_DECREF(dict);
        return nullptr;
    }
    return dict;
}

PyObject* CompiledCountsSampler_get_preprocessing_timing(PyCompiledCountsSampler* self, void*) {
    return timing_to_dict(CompiledCountsSampler_preprocessing_timing_ref(self));
}

PyObject* module_simd_backend(PyObject*, PyObject*) {
    try {
        return PyUnicode_FromString(symft::active_simd_backend().c_str());
    } catch (...) {
        return set_cpp_exception_null();
    }
}

PyObject* module_cuda_enabled(PyObject*, PyObject*) {
#ifdef SYMFT_CPP_ENABLE_CUDA
    Py_RETURN_TRUE;
#else
    Py_RETURN_FALSE;
#endif
}

PyObject* module_active_cuda_backend(PyObject*, PyObject*) {
#ifdef SYMFT_CPP_ENABLE_CUDA
    try {
        return PyUnicode_FromString(symft::cuda::cuda_sampler_backend_name());
    } catch (...) {
        return set_cpp_exception_null();
    }
#else
    return PyUnicode_FromString("unavailable");
#endif
}

PyMethodDef Circuit_methods[] = {
    {
        "compile_sampler",
        reinterpret_cast<PyCFunction>(Circuit_compile_sampler),
        METH_VARARGS | METH_KEYWORDS,
        "compile_sampler($self, /, batch=False, batch_size=0, sample_chunk_shots=0)\n"
        "--\n\n"
        "Compile a reusable measurement sampler.\n\n"
        "A value of 0 selects the runtime default for batch_size or "
        "sample_chunk_shots.",
    },
    {
        "compile_counts_sampler",
        reinterpret_cast<PyCFunction>(Circuit_compile_counts_sampler),
        METH_VARARGS | METH_KEYWORDS,
        "compile_counts_sampler($self, /, batch=True, observable=0, "
        "postselect_detectors=False, batch_size=0, sample_chunk_shots=0, "
        "threads=1, batch_mask_threshold_denominator=2, cuda=False, "
        "cuda_mode='gpu', shots_per_launch=0, threads_per_block=0)\n"
        "--\n\n"
        "Compile a reusable detector/logical-error counts sampler.\n\n"
        "Set cuda=True to use the CUDA counts backend when this extension was "
        "built with CUDA support. Use stream_id when sampling the result to "
        "select a reproducible random stream.",
    },
    {
        "sample",
        reinterpret_cast<PyCFunction>(Circuit_sample),
        METH_VARARGS | METH_KEYWORDS,
        "sample($self, /, shots=1, seed=1, batch=False, bit_packed=False, "
        "batch_size=0, sample_chunk_shots=0)\n"
        "--\n\n"
        "Sample measurement records into a two-dimensional NumPy array.\n\n"
        "The shape is (shots, num_measurements). With bit_packed=True, "
        "the dtype is uint8 and the second dimension is ceil(num_measurements / 8).",
    },
    {
        "sample_counts",
        reinterpret_cast<PyCFunction>(Circuit_sample_counts),
        METH_VARARGS | METH_KEYWORDS,
        "sample_counts($self, /, shots=1, seed=1, batch=True, observable=0, "
        "postselect_detectors=False, batch_size=0, sample_chunk_shots=0, "
        "threads=1, batch_mask_threshold_denominator=2, cuda=False, "
        "cuda_mode='gpu', shots_per_launch=0, threads_per_block=0)\n"
        "--\n\n"
        "Sample detector and logical-observable summary counts.\n\n"
        "Set cuda=True to use the CUDA counts backend when available. Returns "
        "a dict containing shots, discarded, accepted, logical_errors, rates, "
        "active_threads, and timing.",
    },
    {
        "sample_detectors",
        reinterpret_cast<PyCFunction>(Circuit_sample_detectors),
        METH_VARARGS | METH_KEYWORDS,
        "sample_detectors($self, /, shots=1, seed=1, bit_packed=False)\n"
        "--\n\n"
        "Sample detector records into a two-dimensional NumPy array.\n\n"
        "The shape is (shots, num_detectors). With bit_packed=True, "
        "the dtype is uint8 and detector 0 is the low bit of byte 0.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyGetSetDef Circuit_getsets[] = {
    {"num_qubits", reinterpret_cast<getter>(Circuit_get_num_qubits), nullptr, "Number of qubits.", nullptr},
    {
        "num_measurements",
        reinterpret_cast<getter>(Circuit_get_num_measurements),
        nullptr,
        "Number of measurement records.",
        nullptr,
    },
    {"num_detectors", reinterpret_cast<getter>(Circuit_get_num_detectors), nullptr, "Number of detectors.", nullptr},
    {
        "num_observables",
        reinterpret_cast<getter>(Circuit_get_num_observables),
        nullptr,
        "Number of observable indices.",
        nullptr,
    },
    {
        "num_observable_includes",
        reinterpret_cast<getter>(Circuit_get_num_observable_includes),
        nullptr,
        "Number of OBSERVABLE_INCLUDE instructions.",
        nullptr,
    },
    {"detectors", reinterpret_cast<getter>(Circuit_get_detectors), nullptr, "Parsed detector metadata.", nullptr},
    {"observables", reinterpret_cast<getter>(Circuit_get_observables), nullptr, "Parsed observable metadata.", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef CompiledMeasurementSampler_methods[] = {
    {
        "sample",
        reinterpret_cast<PyCFunction>(CompiledMeasurementSampler_sample),
        METH_VARARGS | METH_KEYWORDS,
        "sample($self, /, shots=1, seed=1, bit_packed=False)\n"
        "--\n\n"
        "Sample measurement records using the compiled program.",
    },
    {
        "sample_detectors",
        reinterpret_cast<PyCFunction>(CompiledMeasurementSampler_sample_detectors),
        METH_VARARGS | METH_KEYWORDS,
        "sample_detectors($self, /, shots=1, seed=1, bit_packed=False)\n"
        "--\n\n"
        "Sample detector records using the compiled program.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyGetSetDef CompiledMeasurementSampler_getsets[] = {
    {
        "num_qubits",
        reinterpret_cast<getter>(CompiledMeasurementSampler_get_num_qubits),
        nullptr,
        "Number of qubits.",
        nullptr,
    },
    {
        "num_measurements",
        reinterpret_cast<getter>(CompiledMeasurementSampler_get_num_measurements),
        nullptr,
        "Number of measurement records.",
        nullptr,
    },
    {
        "num_detectors",
        reinterpret_cast<getter>(CompiledMeasurementSampler_get_num_detectors),
        nullptr,
        "Number of detectors.",
        nullptr,
    },
    {
        "max_active_qubits",
        reinterpret_cast<getter>(CompiledMeasurementSampler_get_max_active_qubits),
        nullptr,
        "Maximum active subsystem width.",
        nullptr,
    },
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef CompiledCountsSampler_methods[] = {
    {
        "sample",
        reinterpret_cast<PyCFunction>(CompiledCountsSampler_sample),
        METH_VARARGS | METH_KEYWORDS,
        "sample($self, /, shots=1, stream_id=None)\n"
        "--\n\n"
        "Sample detector and logical-observable summary counts.\n\n"
        "Omit stream_id to advance the sampler's internal stream counter, or "
        "set it explicitly for reproducible results.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyGetSetDef CompiledCountsSampler_getsets[] = {
    {"info", reinterpret_cast<getter>(CompiledCountsSampler_get_info), nullptr, "Sampler metadata.", nullptr},
    {
        "preprocessing_timing",
        reinterpret_cast<getter>(CompiledCountsSampler_get_preprocessing_timing),
        nullptr,
        "Parse/plan timing metadata.",
        nullptr,
    },
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyMethodDef module_methods[] = {
    {
        "simd_backend",
        reinterpret_cast<PyCFunction>(module_simd_backend),
        METH_NOARGS,
        "Return the CPU SIMD backend selected by the C++ runtime.",
    },
    {
        "cuda_enabled",
        reinterpret_cast<PyCFunction>(module_cuda_enabled),
        METH_NOARGS,
        "Return whether this extension was built with CUDA sampler support.",
    },
    {
        "active_cuda_backend",
        reinterpret_cast<PyCFunction>(module_active_cuda_backend),
        METH_NOARGS,
        "Return the CUDA sampler backend name, or 'unavailable' when CUDA support is not built.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "symft._native",
    "Native Python bindings for SymFT.",
    -1,
    module_methods,
};

bool ready_type(PyTypeObject& type) {
    return PyType_Ready(&type) >= 0;
}

} // namespace

PyMODINIT_FUNC PyInit__native() {
    import_array();

    CircuitType.tp_name = "symft.Circuit";
    CircuitType.tp_basicsize = sizeof(PyCircuit);
    CircuitType.tp_itemsize = 0;
    CircuitType.tp_dealloc = reinterpret_cast<destructor>(Circuit_dealloc);
    CircuitType.tp_repr = reinterpret_cast<reprfunc>(Circuit_repr);
    CircuitType.tp_flags = Py_TPFLAGS_DEFAULT;
    CircuitType.tp_doc =
        "Circuit(text=None, path=None)\n"
        "--\n\n"
        "Parsed Stim-style Clifford+T circuit.\n\n"
        "Construct from str or bytes circuit text, or pass an os.PathLike, str, "
        "or bytes object using path. Text and path are mutually exclusive.";
    CircuitType.tp_methods = Circuit_methods;
    CircuitType.tp_getset = Circuit_getsets;
    CircuitType.tp_init = reinterpret_cast<initproc>(Circuit_init);
    CircuitType.tp_new = PyType_GenericNew;

    CompiledMeasurementSamplerType.tp_name = "symft.CompiledMeasurementSampler";
    CompiledMeasurementSamplerType.tp_basicsize = sizeof(PyCompiledMeasurementSampler);
    CompiledMeasurementSamplerType.tp_itemsize = 0;
    CompiledMeasurementSamplerType.tp_dealloc =
        reinterpret_cast<destructor>(CompiledMeasurementSampler_dealloc);
    CompiledMeasurementSamplerType.tp_repr =
        reinterpret_cast<reprfunc>(CompiledMeasurementSampler_repr);
    CompiledMeasurementSamplerType.tp_flags = Py_TPFLAGS_DEFAULT;
    CompiledMeasurementSamplerType.tp_doc = "Reusable SymFT measurement sampler.";
    CompiledMeasurementSamplerType.tp_methods = CompiledMeasurementSampler_methods;
    CompiledMeasurementSamplerType.tp_getset = CompiledMeasurementSampler_getsets;

    CompiledCountsSamplerType.tp_name = "symft.CompiledCountsSampler";
    CompiledCountsSamplerType.tp_basicsize = sizeof(PyCompiledCountsSampler);
    CompiledCountsSamplerType.tp_itemsize = 0;
    CompiledCountsSamplerType.tp_dealloc =
        reinterpret_cast<destructor>(CompiledCountsSampler_dealloc);
    CompiledCountsSamplerType.tp_repr =
        reinterpret_cast<reprfunc>(CompiledCountsSampler_repr);
    CompiledCountsSamplerType.tp_flags = Py_TPFLAGS_DEFAULT;
    CompiledCountsSamplerType.tp_doc = "Reusable SymFT detector/logical-error counts sampler.";
    CompiledCountsSamplerType.tp_methods = CompiledCountsSampler_methods;
    CompiledCountsSamplerType.tp_getset = CompiledCountsSampler_getsets;

    if (!ready_type(CircuitType) ||
        !ready_type(CompiledMeasurementSamplerType) ||
        !ready_type(CompiledCountsSamplerType)) {
        return nullptr;
    }

    PyObject* module = PyModule_Create(&module_def);
    if (module == nullptr) {
        return nullptr;
    }

    SymFTError = PyErr_NewException("symft.SymFTError", PyExc_RuntimeError, nullptr);
    if (SymFTError == nullptr) {
        Py_DECREF(module);
        return nullptr;
    }

    Py_INCREF(&CircuitType);
    Py_INCREF(&CompiledMeasurementSamplerType);
    Py_INCREF(&CompiledCountsSamplerType);
    Py_INCREF(SymFTError);

    PyModule_AddObject(module, "Circuit", reinterpret_cast<PyObject*>(&CircuitType));
    PyModule_AddObject(
        module,
        "CompiledMeasurementSampler",
        reinterpret_cast<PyObject*>(&CompiledMeasurementSamplerType));
    PyModule_AddObject(
        module,
        "CompiledCountsSampler",
        reinterpret_cast<PyObject*>(&CompiledCountsSamplerType));
    PyModule_AddObject(module, "SymFTError", SymFTError);

    return module;
}
