#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numpy/arrayobject.h>
#include "stabilizers.h"

#include <cuda_runtime.h>
#include <simulator.hpp>
#include <functional>
#include "write.hpp"

#include <climits>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <cmath>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

PyObject* SoftCudaError = nullptr;

struct AllowThreads {
    PyThreadState* state;

    AllowThreads() : state(PyEval_SaveThread()) {}
    ~AllowThreads() { PyEval_RestoreThread(state); }

    AllowThreads(const AllowThreads&) = delete;
    AllowThreads& operator=(const AllowThreads&) = delete;
};

struct PyCpuCircuit {
    PyObject_HEAD
    CircuitInput* input;
    int num_qubits_override;
};

PyTypeObject CpuCircuitType = {PyVarObject_HEAD_INIT(nullptr, 0)};

int set_cpu_exception() {
    try {
        throw;
    } catch (const std::exception& ex) {
        PyErr_SetString(SoftCudaError, ex.what());
    } catch (...) {
        PyErr_SetString(SoftCudaError, "unknown SOFT CPU exception");
    }
    return -1;
}

PyObject* set_cpu_exception_null() {
    set_cpu_exception();
    return nullptr;
}

bool object_to_string(PyObject* object, std::string& out, const char* name) {
    if (PyUnicode_Check(object)) {
        Py_ssize_t size = 0;
        const char* data = PyUnicode_AsUTF8AndSize(object, &size);
        if (data == nullptr) {
            return false;
        }
        out.assign(data, static_cast<size_t>(size));
        return true;
    }
    if (PyBytes_Check(object)) {
        char* data = nullptr;
        Py_ssize_t size = 0;
        if (PyBytes_AsStringAndSize(object, &data, &size) < 0) {
            return false;
        }
        out.assign(data, static_cast<size_t>(size));
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
    const int status = PyDict_SetItemString(dict, key, value);
    Py_DECREF(value);
    return status == 0;
}

int cpu_count_gate(const CircuitInput& input, const char* name) {
    int count = 0;
    for (const auto& op : input.circuit) {
        if (op.gate == name) {
            count++;
        }
    }
    return count;
}

int cpu_num_measurements(const CircuitInput& input) {
    int count = 0;
    for (const auto& op : input.circuit) {
        if (op.gate == "M" || op.gate == "MPP") {
            count++;
        }
    }
    return count;
}

int cpu_num_observables(const CircuitInput& input) {
    int max_index = -1;
    for (const auto& op : input.circuit) {
        if (op.gate != "OBSERVABLE_INCLUDE") {
            continue;
        }
        const int index = op.args.empty() ? 0 : static_cast<int>(op.args[0]);
        if (index > max_index) {
            max_index = index;
        }
    }
    return max_index + 1;
}

int resolve_cpu_num_qubits(PyCpuCircuit* self, int override_num_qubits, int& out) {
    if (self->input == nullptr) {
        PyErr_SetString(SoftCudaError, "CPU circuit is closed");
        return -1;
    }
    out = self->input->num_qubits;
    if (self->num_qubits_override > 0) {
        out = self->num_qubits_override;
    }
    if (override_num_qubits > 0) {
        out = override_num_qubits;
    }
    if (out < self->input->num_qubits) {
        PyErr_SetString(PyExc_ValueError, "num_qubits is smaller than the largest qubit referenced by the circuit");
        return -1;
    }
    return 0;
}

PyObject* CpuCircuit_new(PyTypeObject* type, PyObject*, PyObject*) {
    PyCpuCircuit* self = reinterpret_cast<PyCpuCircuit*>(type->tp_alloc(type, 0));
    if (self != nullptr) {
        self->input = nullptr;
        self->num_qubits_override = -1;
    }
    return reinterpret_cast<PyObject*>(self);
}

int CpuCircuit_init(PyCpuCircuit* self, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"text", "path", "num_qubits", nullptr};
    PyObject* text_object = Py_None;
    PyObject* path_object = Py_None;
    int num_qubits = -1;
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|OOi:CpuCircuit",
            const_cast<char**>(keywords),
            &text_object,
            &path_object,
            &num_qubits)) {
        return -1;
    }
    if (text_object != Py_None && path_object != Py_None) {
        PyErr_SetString(PyExc_TypeError, "pass either text or path, not both");
        return -1;
    }
    if (num_qubits == 0 || num_qubits < -1) {
        PyErr_SetString(PyExc_ValueError, "num_qubits must be positive or -1");
        return -1;
    }

    CircuitInput parsed;
    try {
        if (path_object != Py_None) {
            std::string path;
            if (!path_to_string(path_object, path)) {
                return -1;
            }
            parsed = load_stim_file(path);
        } else {
            std::string text;
            if (text_object != Py_None && !object_to_string(text_object, text, "text")) {
                return -1;
            }
            parsed = load_stim_text(text);
        }
    } catch (...) {
        return set_cpu_exception();
    }

    if (num_qubits > 0 && num_qubits < parsed.num_qubits) {
        PyErr_SetString(PyExc_ValueError, "num_qubits is smaller than the largest qubit referenced by the circuit");
        return -1;
    }
    delete self->input;
    try {
        self->input = new CircuitInput(std::move(parsed));
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return -1;
    }
    self->num_qubits_override = num_qubits;
    return 0;
}

void CpuCircuit_dealloc(PyCpuCircuit* self) {
    delete self->input;
    self->input = nullptr;
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* CpuCircuit_sample_matrix(PyCpuCircuit* self, PyObject* args, PyObject* kwargs, int kind) {
    static const char* keywords[] = {"shots", "num_qubits", "bit_packed", nullptr};
    long long shots = 1;
    int num_qubits_override = -1;
    int bit_packed = 0;
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|Lip:sample_matrix",
            const_cast<char**>(keywords),
            &shots,
            &num_qubits_override,
            &bit_packed)) {
        return nullptr;
    }
    if (shots < 0) {
        PyErr_SetString(PyExc_ValueError, "shots must be nonnegative");
        return nullptr;
    }
    if (shots > static_cast<long long>(std::numeric_limits<npy_intp>::max())) {
        PyErr_SetString(PyExc_OverflowError, "shots exceeds NumPy dimension limits");
        return nullptr;
    }
    int num_qubits = 0;
    if (resolve_cpu_num_qubits(self, num_qubits_override, num_qubits) < 0) {
        return nullptr;
    }

    const int bits = kind == 0 ? cpu_num_measurements(*self->input) : cpu_count_gate(*self->input, "DETECTOR");
    const int columns = bit_packed ? (bits + 7) / 8 : bits;
    npy_intp dims[2] = {static_cast<npy_intp>(shots), static_cast<npy_intp>(columns)};
    PyObject* array = PyArray_SimpleNew(2, dims, bit_packed ? NPY_UINT8 : NPY_BOOL);
    if (array == nullptr) {
        return nullptr;
    }
    const size_t item_bytes = bit_packed ? sizeof(unsigned char) : sizeof(npy_bool);
    std::memset(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)), 0, static_cast<size_t>(dims[0]) * static_cast<size_t>(dims[1]) * item_bytes);

    try {
        for (long long shot = 0; shot < shots; shot++) {
            Stabilizer stabilizer(num_qubits);
            SimulationResult result = stabilizer.run(self->input->circuit, false, false);
            if (kind == 0) {
                for (int i = 0; i < bits && i < static_cast<int>(result.measurements.size()); i++) {
                    const bool value = result.measurements[i].reg != 0;
                    if (!value) {
                        continue;
                    }
                    if (bit_packed) {
                        unsigned char* data = static_cast<unsigned char*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)));
                        data[shot * columns + i / 8] |= static_cast<unsigned char>(1u << (i & 7));
                    } else {
                        npy_bool* data = static_cast<npy_bool*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)));
                        data[shot * columns + i] = NPY_TRUE;
                    }
                }
            } else {
                for (int i = 0; i < bits && i < static_cast<int>(result.detectors.size()); i++) {
                    const bool value = result.detectors[i].value;
                    if (!value) {
                        continue;
                    }
                    if (bit_packed) {
                        unsigned char* data = static_cast<unsigned char*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)));
                        data[shot * columns + i / 8] |= static_cast<unsigned char>(1u << (i & 7));
                    } else {
                        npy_bool* data = static_cast<npy_bool*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)));
                        data[shot * columns + i] = NPY_TRUE;
                    }
                }
            }
        }
    } catch (...) {
        Py_DECREF(array);
        return set_cpu_exception_null();
    }
    return array;
}

PyObject* CpuCircuit_sample_measurements(PyCpuCircuit* self, PyObject* args, PyObject* kwargs) {
    return CpuCircuit_sample_matrix(self, args, kwargs, 0);
}

PyObject* CpuCircuit_sample_detectors(PyCpuCircuit* self, PyObject* args, PyObject* kwargs) {
    return CpuCircuit_sample_matrix(self, args, kwargs, 1);
}

double safe_ratio(long long numerator, long long denominator) {
    if (denominator == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

PyObject* CpuCircuit_sample_counts(PyCpuCircuit* self, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {"shots", "observable", "num_qubits", nullptr};
    long long shots = 1;
    int observable = 0;
    int num_qubits_override = -1;
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|Lii:sample_counts",
            const_cast<char**>(keywords),
            &shots,
            &observable,
            &num_qubits_override)) {
        return nullptr;
    }
    if (shots < 0) {
        PyErr_SetString(PyExc_ValueError, "shots must be nonnegative");
        return nullptr;
    }
    if (observable < 0) {
        PyErr_SetString(PyExc_ValueError, "observable must be nonnegative");
        return nullptr;
    }
    int num_qubits = 0;
    if (resolve_cpu_num_qubits(self, num_qubits_override, num_qubits) < 0) {
        return nullptr;
    }

    const auto sample_start = std::chrono::steady_clock::now();
    long long discarded = 0;
    long long logical_errors = 0;
    try {
        for (long long shot = 0; shot < shots; shot++) {
            Stabilizer stabilizer(num_qubits);
            SimulationResult result = stabilizer.run(self->input->circuit, false, false);
            if (result.discarded) {
                discarded++;
                continue;
            }
            if (observable < static_cast<int>(result.observable_bits.size()) && result.observable_bits[observable] != 0) {
                logical_errors++;
            }
        }
    } catch (...) {
        return set_cpu_exception_null();
    }

    const auto sample_end = std::chrono::steady_clock::now();
    const double execute_s = std::chrono::duration<double>(sample_end - sample_start).count();
    const double sample_s = execute_s;
    const long long accepted = shots - discarded;
    PyObject* dict = PyDict_New();
    if (dict == nullptr) {
        return nullptr;
    }
    PyObject* timing = PyDict_New();
    if (timing == nullptr) {
        Py_DECREF(dict);
        return nullptr;
    }
    if (!dict_set_owned(timing, "parse_s", PyFloat_FromDouble(0.0)) ||
        !dict_set_owned(timing, "plan_s", PyFloat_FromDouble(0.0)) ||
        !dict_set_owned(timing, "presample_s", PyFloat_FromDouble(0.0)) ||
        !dict_set_owned(timing, "execute_s", PyFloat_FromDouble(execute_s)) ||
        !dict_set_owned(timing, "accumulate_s", PyFloat_FromDouble(0.0)) ||
        !dict_set_owned(timing, "sample_s", PyFloat_FromDouble(sample_s))) {
        Py_DECREF(timing);
        Py_DECREF(dict);
        return nullptr;
    }
    if (!dict_set_owned(dict, "shots", PyLong_FromLongLong(shots)) ||
        !dict_set_owned(dict, "discarded", PyLong_FromLongLong(discarded)) ||
        !dict_set_owned(dict, "accepted", PyLong_FromLongLong(accepted)) ||
        !dict_set_owned(dict, "logical_errors", PyLong_FromLongLong(logical_errors)) ||
        !dict_set_owned(dict, "discard_rate", PyFloat_FromDouble(safe_ratio(discarded, shots))) ||
        !dict_set_owned(dict, "logical_error_rate", PyFloat_FromDouble(safe_ratio(logical_errors, accepted))) ||
        !dict_set_owned(dict, "active_threads", PyLong_FromLong(1)) ||
        !dict_set_owned(dict, "backend", PyUnicode_FromString("cpu"))) {
        Py_DECREF(timing);
        Py_DECREF(dict);
        return nullptr;
    }
    if (PyDict_SetItemString(dict, "timing", timing) < 0) {
        Py_DECREF(timing);
        Py_DECREF(dict);
        return nullptr;
    }
    Py_DECREF(timing);
    return dict;
}

PyObject* CpuCircuit_get_num_qubits(PyCpuCircuit* self, void*) {
    int num_qubits = 0;
    if (resolve_cpu_num_qubits(self, -1, num_qubits) < 0) {
        return nullptr;
    }
    return PyLong_FromLong(num_qubits);
}

PyObject* CpuCircuit_get_num_measurements(PyCpuCircuit* self, void*) {
    if (self->input == nullptr) {
        PyErr_SetString(SoftCudaError, "CPU circuit is closed");
        return nullptr;
    }
    return PyLong_FromLong(cpu_num_measurements(*self->input));
}

PyObject* CpuCircuit_get_num_detectors(PyCpuCircuit* self, void*) {
    if (self->input == nullptr) {
        PyErr_SetString(SoftCudaError, "CPU circuit is closed");
        return nullptr;
    }
    return PyLong_FromLong(cpu_count_gate(*self->input, "DETECTOR"));
}

PyObject* CpuCircuit_get_num_observables(PyCpuCircuit* self, void*) {
    if (self->input == nullptr) {
        PyErr_SetString(SoftCudaError, "CPU circuit is closed");
        return nullptr;
    }
    return PyLong_FromLong(cpu_num_observables(*self->input));
}

PyMethodDef CpuCircuit_methods[] = {
    {"sample", reinterpret_cast<PyCFunction>(CpuCircuit_sample_measurements), METH_VARARGS | METH_KEYWORDS, "Sample measurement records with the CPU stabilizer simulator."},
    {"sample_measurements", reinterpret_cast<PyCFunction>(CpuCircuit_sample_measurements), METH_VARARGS | METH_KEYWORDS, "Sample measurement records with the CPU stabilizer simulator."},
    {"sample_detectors", reinterpret_cast<PyCFunction>(CpuCircuit_sample_detectors), METH_VARARGS | METH_KEYWORDS, "Sample detector records with the CPU stabilizer simulator."},
    {"sample_counts", reinterpret_cast<PyCFunction>(CpuCircuit_sample_counts), METH_VARARGS | METH_KEYWORDS, "Return detector discard and logical error counts from CPU sampling."},
    {nullptr, nullptr, 0, nullptr},
};

PyGetSetDef CpuCircuit_getset[] = {
    {"num_qubits", reinterpret_cast<getter>(CpuCircuit_get_num_qubits), nullptr, "Number of qubits.", nullptr},
    {"num_measurements", reinterpret_cast<getter>(CpuCircuit_get_num_measurements), nullptr, "Number of measurements.", nullptr},
    {"num_detectors", reinterpret_cast<getter>(CpuCircuit_get_num_detectors), nullptr, "Number of detectors.", nullptr},
    {"num_observables", reinterpret_cast<getter>(CpuCircuit_get_num_observables), nullptr, "Number of observables.", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

int ready_cpu_circuit_type() {
    CpuCircuitType.tp_name = "soft._native.CpuCircuit";
    CpuCircuitType.tp_basicsize = sizeof(PyCpuCircuit);
    CpuCircuitType.tp_itemsize = 0;
    CpuCircuitType.tp_dealloc = reinterpret_cast<destructor>(CpuCircuit_dealloc);
    CpuCircuitType.tp_flags = Py_TPFLAGS_DEFAULT;
    CpuCircuitType.tp_doc = "SOFT CPU stabilizer circuit.";
    CpuCircuitType.tp_methods = CpuCircuit_methods;
    CpuCircuitType.tp_getset = CpuCircuit_getset;
    CpuCircuitType.tp_init = reinterpret_cast<initproc>(CpuCircuit_init);
    CpuCircuitType.tp_new = CpuCircuit_new;
    return PyType_Ready(&CpuCircuitType);
}

struct PySimulator {
    PyObject_HEAD
    SoftCuda::Simulator* simulator;
    SoftCuda::SimulatorArgs args;
};

PyTypeObject SimulatorType = {PyVarObject_HEAD_INIT(nullptr, 0)};

int set_cuda_error(cudaError_t error, const char* context) {
    if (error == cudaSuccess) {
        return 0;
    }
    PyErr_Format(
        SoftCudaError,
        "%s failed: %s: %s",
        context,
        cudaGetErrorName(error),
        cudaGetErrorString(error));
    return -1;
}

PyObject* set_cuda_error_null(cudaError_t error, const char* context) {
    set_cuda_error(error, context);
    return nullptr;
}

int require_simulator(PySimulator* self) {
    if (self->simulator == nullptr || self->simulator->shots_state_ptr.ptr == nullptr) {
        PyErr_SetString(SoftCudaError, "Simulator is closed");
        return -1;
    }
    return 0;
}

bool check_qubit(PySimulator* self, SoftCuda::Qid target, const char* name) {
    if (target >= self->args.qubits_n) {
        PyErr_Format(
            PyExc_IndexError,
            "%s target %u is out of range for %u qubits",
            name,
            target,
            self->args.qubits_n);
        return false;
    }
    return true;
}

bool check_probability(double probability, const char* name) {
    if (!(probability >= 0.0 && probability <= 1.0)) {
        PyErr_Format(PyExc_ValueError, "%s probability must be between 0 and 1", name);
        return false;
    }
    return true;
}

bool parse_bit(int value, SoftCuda::Bit& out, const char* name) {
    if (value != 0 && value != 1) {
        PyErr_Format(PyExc_ValueError, "%s must be 0 or 1", name);
        return false;
    }
    out = value != 0;
    return true;
}

bool parse_mid(PyObject* object, SoftCuda::Mid& out, const char* name) {
    const unsigned long value = PyLong_AsUnsignedLong(object);
    if (PyErr_Occurred()) {
        return false;
    }
    if (value > std::numeric_limits<SoftCuda::Mid>::max()) {
        PyErr_Format(PyExc_OverflowError, "%s exceeds the supported uint range", name);
        return false;
    }
    out = static_cast<SoftCuda::Mid>(value);
    return true;
}

bool parse_int(PyObject* object, SoftCuda::Int& out, const char* name) {
    const long value = PyLong_AsLong(object);
    if (PyErr_Occurred()) {
        return false;
    }
    if (value < std::numeric_limits<SoftCuda::Int>::min() ||
        value > std::numeric_limits<SoftCuda::Int>::max()) {
        PyErr_Format(PyExc_OverflowError, "%s exceeds the supported int range", name);
        return false;
    }
    out = static_cast<SoftCuda::Int>(value);
    return true;
}

template<SoftCuda::Mid m>
bool fill_reduce_args(
    PyObject* pointers_object,
    SoftCuda::ClassicalReduceArgs<m>& out,
    SoftCuda::Mid max_pointer,
    const char* name) {
    PyObject* pointers = PySequence_Fast(pointers_object, "pointers must be a sequence");
    if (pointers == nullptr) {
        return false;
    }

    const Py_ssize_t size = PySequence_Fast_GET_SIZE(pointers);
    if (size > static_cast<Py_ssize_t>(m)) {
        Py_DECREF(pointers);
        PyErr_Format(PyExc_ValueError, "%s accepts at most %u pointers", name, m);
        return false;
    }

    out.n = static_cast<SoftCuda::Mid>(size);
    PyObject** items = PySequence_Fast_ITEMS(pointers);
    for (Py_ssize_t i = 0; i < size; ++i) {
        SoftCuda::Mid pointer = 0;
        if (!parse_mid(items[i], pointer, "pointer")) {
            Py_DECREF(pointers);
            return false;
        }
        if (pointer >= max_pointer) {
            Py_DECREF(pointers);
            PyErr_Format(
                PyExc_IndexError,
                "%s pointer %u is out of range for %u memory integers",
                name,
                pointer,
                max_pointer);
            return false;
        }
        out.pointers.get(static_cast<unsigned int>(i)) = pointer;
    }

    Py_DECREF(pointers);
    return true;
}

template<SoftCuda::Mid m>
bool fill_match_args(
    PyObject* pointers_object,
    PyObject* values_object,
    SoftCuda::ClassicalMatchArgs<m>& out,
    SoftCuda::Mid max_pointer) {
    PyObject* pointers = PySequence_Fast(pointers_object, "pointers must be a sequence");
    if (pointers == nullptr) {
        return false;
    }
    PyObject* values = PySequence_Fast(values_object, "values must be a sequence");
    if (values == nullptr) {
        Py_DECREF(pointers);
        return false;
    }

    const Py_ssize_t pointers_size = PySequence_Fast_GET_SIZE(pointers);
    const Py_ssize_t values_size = PySequence_Fast_GET_SIZE(values);
    if (pointers_size != values_size) {
        Py_DECREF(values);
        Py_DECREF(pointers);
        PyErr_SetString(PyExc_ValueError, "pointers and values must have the same length");
        return false;
    }
    if (pointers_size > static_cast<Py_ssize_t>(m)) {
        Py_DECREF(values);
        Py_DECREF(pointers);
        PyErr_Format(PyExc_ValueError, "MATCH accepts at most %u pointer/value pairs", m);
        return false;
    }

    out.n = static_cast<SoftCuda::Mid>(pointers_size);
    PyObject** pointer_items = PySequence_Fast_ITEMS(pointers);
    PyObject** value_items = PySequence_Fast_ITEMS(values);
    for (Py_ssize_t i = 0; i < pointers_size; ++i) {
        SoftCuda::Mid pointer = 0;
        SoftCuda::Int value = 0;
        if (!parse_mid(pointer_items[i], pointer, "pointer") ||
            !parse_int(value_items[i], value, "value")) {
            Py_DECREF(values);
            Py_DECREF(pointers);
            return false;
        }
        if (pointer >= max_pointer) {
            Py_DECREF(values);
            Py_DECREF(pointers);
            PyErr_Format(
                PyExc_IndexError,
                "MATCH pointer %u is out of range for %u memory integers",
                pointer,
                max_pointer);
            return false;
        }
        out.pointers.get(static_cast<unsigned int>(i)) = pointer;
        out.values.get(static_cast<unsigned int>(i)) = value;
    }

    Py_DECREF(values);
    Py_DECREF(pointers);
    return true;
}

int synchronize(PySimulator* self, const char* context) {
    if (require_simulator(self) < 0) {
        return -1;
    }
    cudaError_t error = cudaSuccess;
    {
        AllowThreads allow;
        error = cudaStreamSynchronize(self->simulator->stream);
    }
    return set_cuda_error(error, context);
}

int check_launch(const char* context) {
    return set_cuda_error(cudaGetLastError(), context);
}

template<typename T, typename PointerGetter>
PyObject* copy_per_shot(
    PySimulator* self,
    int numpy_type,
    const char* context,
    PointerGetter get_pointer) {
    if (require_simulator(self) < 0) {
        return nullptr;
    }

    const SoftCuda::ShotsStatePtr shots = self->simulator->shots_state_ptr;
    npy_intp dims[1] = {static_cast<npy_intp>(shots.shots_n)};
    PyObject* array = PyArray_SimpleNew(1, dims, numpy_type);
    if (array == nullptr) {
        return nullptr;
    }

    const size_t pitch = shots.get_shot_size_bytes_n() + shots.get_shot_pad_bytes_n();
    const void* src = static_cast<const void*>(get_pointer(shots.get_shot_ptr(0)));
    cudaError_t error = cudaSuccess;
    {
        AllowThreads allow;
        error = cudaMemcpy2DAsync(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)),
            sizeof(T),
            src,
            pitch,
            sizeof(T),
            shots.shots_n,
            cudaMemcpyDeviceToHost,
            self->simulator->stream);
        if (error == cudaSuccess) {
            error = cudaStreamSynchronize(self->simulator->stream);
        }
    }
    if (set_cuda_error(error, context) < 0) {
        Py_DECREF(array);
        return nullptr;
    }
    return array;
}

template<typename T, typename PointerGetter>
PyObject* copy_memory_matrix(
    PySimulator* self,
    SoftCuda::Mid width_items,
    int numpy_type,
    const char* context,
    PointerGetter get_pointer) {
    if (require_simulator(self) < 0) {
        return nullptr;
    }

    const SoftCuda::ShotsStatePtr shots = self->simulator->shots_state_ptr;
    npy_intp dims[2] = {
        static_cast<npy_intp>(shots.shots_n),
        static_cast<npy_intp>(width_items),
    };
    PyObject* array = PyArray_SimpleNew(2, dims, numpy_type);
    if (array == nullptr) {
        return nullptr;
    }
    if (width_items == 0) {
        return array;
    }

    const size_t pitch = shots.get_shot_size_bytes_n() + shots.get_shot_pad_bytes_n();
    const void* src = static_cast<const void*>(get_pointer(shots.get_shot_ptr(0)));
    cudaError_t error = cudaSuccess;
    {
        AllowThreads allow;
        error = cudaMemcpy2DAsync(
            PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)),
            width_items * sizeof(T),
            src,
            pitch,
            width_items * sizeof(T),
            shots.shots_n,
            cudaMemcpyDeviceToHost,
            self->simulator->stream);
        if (error == cudaSuccess) {
            error = cudaStreamSynchronize(self->simulator->stream);
        }
    }
    if (set_cuda_error(error, context) < 0) {
        Py_DECREF(array);
        return nullptr;
    }
    return array;
}

void destroy_simulator(PySimulator* self) {
    if (self->simulator != nullptr) {
        self->simulator->destroy();
        delete self->simulator;
        self->simulator = nullptr;
    }
}

PyObject* Simulator_new(PyTypeObject* type, PyObject*, PyObject*) {
    PySimulator* self = reinterpret_cast<PySimulator*>(type->tp_alloc(type, 0));
    if (self != nullptr) {
        self->simulator = nullptr;
        self->args = SoftCuda::SimulatorArgs{};
    }
    return reinterpret_cast<PyObject*>(self);
}

int Simulator_init(PySimulator* self, PyObject* args, PyObject* kwargs) {
    static const char* keywords[] = {
        "shot_i",
        "shots_n",
        "qubits_n",
        "entries_m",
        "mem_ints_n",
        "mem_flts_n",
        "epsilon",
        "seed",
        nullptr,
    };

    SoftCuda::SimulatorArgs parsed{};
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "|IIIIIIdK:Simulator",
            const_cast<char**>(keywords),
            &parsed.shot_i,
            &parsed.shots_n,
            &parsed.qubits_n,
            &parsed.entries_m,
            &parsed.mem_ints_n,
            &parsed.mem_flts_n,
            &parsed.epsilon,
            &parsed.seed)) {
        return -1;
    }

    if (parsed.shots_n == 0) {
        PyErr_SetString(PyExc_ValueError, "shots_n must be positive");
        return -1;
    }
    if (parsed.qubits_n == 0 || parsed.qubits_n > 64) {
        PyErr_SetString(PyExc_ValueError, "qubits_n must be between 1 and 64");
        return -1;
    }
    if (parsed.entries_m == 0) {
        PyErr_SetString(PyExc_ValueError, "entries_m must be positive");
        return -1;
    }

    destroy_simulator(self);
    self->args = parsed;
    try {
        self->simulator = new SoftCuda::Simulator();
    } catch (const std::bad_alloc&) {
        PyErr_NoMemory();
        return -1;
    }

    cudaError_t error = cudaSuccess;
    {
        AllowThreads allow;
        error = self->simulator->create(parsed);
    }
    if (set_cuda_error(error, "Simulator.create") < 0) {
        destroy_simulator(self);
        return -1;
    }
    return 0;
}

void Simulator_dealloc(PySimulator* self) {
    destroy_simulator(self);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

PyObject* Simulator_close(PySimulator* self, PyObject*) {
    destroy_simulator(self);
    Py_RETURN_NONE;
}

PyObject* Simulator_synchronize(PySimulator* self, PyObject*) {
    if (synchronize(self, "Simulator.synchronize") < 0) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

using QidOp = void (SoftCuda::Simulator::*)(SoftCuda::Qid) const;
using QidQidOp = void (SoftCuda::Simulator::*)(SoftCuda::Qid, SoftCuda::Qid) const;
using MeasureValueOp = void (SoftCuda::Simulator::*)(SoftCuda::Qid, SoftCuda::Bit) const;
using Noise1Op = void (SoftCuda::Simulator::*)(SoftCuda::Flt, SoftCuda::Qid) const;
using Noise2Op = void (SoftCuda::Simulator::*)(SoftCuda::Flt, SoftCuda::Qid, SoftCuda::Qid) const;

PyObject* call_qid_op(PySimulator* self, PyObject* args, QidOp op, const char* name) {
    if (require_simulator(self) < 0) {
        return nullptr;
    }
    SoftCuda::Qid target = 0;
    if (!PyArg_ParseTuple(args, "I", &target)) {
        return nullptr;
    }
    if (!check_qubit(self, target, name)) {
        return nullptr;
    }
    {
        AllowThreads allow;
        (self->simulator->*op)(target);
    }
    if (check_launch(name) < 0) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* call_qid_qid_op(PySimulator* self, PyObject* args, QidQidOp op, const char* name) {
    if (require_simulator(self) < 0) {
        return nullptr;
    }
    SoftCuda::Qid control = 0;
    SoftCuda::Qid target = 0;
    if (!PyArg_ParseTuple(args, "II", &control, &target)) {
        return nullptr;
    }
    if (!check_qubit(self, control, name) || !check_qubit(self, target, name)) {
        return nullptr;
    }
    {
        AllowThreads allow;
        (self->simulator->*op)(control, target);
    }
    if (check_launch(name) < 0) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* call_measure_value_op(
    PySimulator* self,
    PyObject* args,
    MeasureValueOp op,
    const char* name) {
    if (require_simulator(self) < 0) {
        return nullptr;
    }
    SoftCuda::Qid target = 0;
    int value_int = 0;
    if (!PyArg_ParseTuple(args, "Ii", &target, &value_int)) {
        return nullptr;
    }
    SoftCuda::Bit value = false;
    if (!check_qubit(self, target, name) || !parse_bit(value_int, value, "value")) {
        return nullptr;
    }
    {
        AllowThreads allow;
        (self->simulator->*op)(target, value);
    }
    if (check_launch(name) < 0) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* call_noise1_op(PySimulator* self, PyObject* args, Noise1Op op, const char* name) {
    if (require_simulator(self) < 0) {
        return nullptr;
    }
    double probability = 0.0;
    SoftCuda::Qid target = 0;
    if (!PyArg_ParseTuple(args, "dI", &probability, &target)) {
        return nullptr;
    }
    if (!check_probability(probability, name) || !check_qubit(self, target, name)) {
        return nullptr;
    }
    {
        AllowThreads allow;
        (self->simulator->*op)(static_cast<SoftCuda::Flt>(probability), target);
    }
    if (check_launch(name) < 0) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* call_noise2_op(PySimulator* self, PyObject* args, Noise2Op op, const char* name) {
    if (require_simulator(self) < 0) {
        return nullptr;
    }
    double probability = 0.0;
    SoftCuda::Qid target0 = 0;
    SoftCuda::Qid target1 = 0;
    if (!PyArg_ParseTuple(args, "dII", &probability, &target0, &target1)) {
        return nullptr;
    }
    if (!check_probability(probability, name) ||
        !check_qubit(self, target0, name) ||
        !check_qubit(self, target1, name)) {
        return nullptr;
    }
    {
        AllowThreads allow;
        (self->simulator->*op)(static_cast<SoftCuda::Flt>(probability), target0, target1);
    }
    if (check_launch(name) < 0) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* Simulator_apply_x(PySimulator* self, PyObject* args) {
    return call_qid_op(self, args, &SoftCuda::Simulator::apply_x, "apply_x");
}

PyObject* Simulator_apply_y(PySimulator* self, PyObject* args) {
    return call_qid_op(self, args, &SoftCuda::Simulator::apply_y, "apply_y");
}

PyObject* Simulator_apply_z(PySimulator* self, PyObject* args) {
    return call_qid_op(self, args, &SoftCuda::Simulator::apply_z, "apply_z");
}

PyObject* Simulator_apply_h(PySimulator* self, PyObject* args) {
    return call_qid_op(self, args, &SoftCuda::Simulator::apply_h, "apply_h");
}

PyObject* Simulator_apply_s(PySimulator* self, PyObject* args) {
    return call_qid_op(self, args, &SoftCuda::Simulator::apply_s, "apply_s");
}

PyObject* Simulator_apply_sdg(PySimulator* self, PyObject* args) {
    return call_qid_op(self, args, &SoftCuda::Simulator::apply_sdg, "apply_sdg");
}

PyObject* Simulator_apply_t(PySimulator* self, PyObject* args) {
    return call_qid_op(self, args, &SoftCuda::Simulator::apply_t, "apply_t");
}

PyObject* Simulator_apply_tdg(PySimulator* self, PyObject* args) {
    return call_qid_op(self, args, &SoftCuda::Simulator::apply_tdg, "apply_tdg");
}

PyObject* Simulator_apply_cx(PySimulator* self, PyObject* args) {
    return call_qid_qid_op(self, args, &SoftCuda::Simulator::apply_cx, "apply_cx");
}

PyObject* Simulator_apply_measure(PySimulator* self, PyObject* args) {
    return call_qid_op(self, args, &SoftCuda::Simulator::apply_measure, "apply_measure");
}

PyObject* Simulator_apply_desire(PySimulator* self, PyObject* args) {
    return call_measure_value_op(self, args, &SoftCuda::Simulator::apply_desire, "apply_desire");
}

PyObject* Simulator_apply_reset(PySimulator* self, PyObject* args) {
    return call_measure_value_op(self, args, &SoftCuda::Simulator::apply_reset, "apply_reset");
}

PyObject* Simulator_apply_noise_x(PySimulator* self, PyObject* args) {
    return call_noise1_op(self, args, &SoftCuda::Simulator::apply_noise_x, "apply_noise_x");
}

PyObject* Simulator_apply_noise_z(PySimulator* self, PyObject* args) {
    return call_noise1_op(self, args, &SoftCuda::Simulator::apply_noise_z, "apply_noise_z");
}

PyObject* Simulator_apply_noise_depo1(PySimulator* self, PyObject* args) {
    return call_noise1_op(self, args, &SoftCuda::Simulator::apply_noise_depo1, "apply_noise_depo1");
}

PyObject* Simulator_apply_noise_depo2(PySimulator* self, PyObject* args) {
    return call_noise2_op(self, args, &SoftCuda::Simulator::apply_noise_depo2, "apply_noise_depo2");
}

PyObject* Simulator_apply_classical_flip(PySimulator* self, PyObject*) {
    if (require_simulator(self) < 0) {
        return nullptr;
    }
    {
        AllowThreads allow;
        self->simulator->apply_classical_flip();
    }
    if (check_launch("apply_classical_flip") < 0) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* Simulator_apply_classical_random_flip(PySimulator* self, PyObject* args) {
    if (require_simulator(self) < 0) {
        return nullptr;
    }
    double probability = 0.0;
    if (!PyArg_ParseTuple(args, "d", &probability)) {
        return nullptr;
    }
    if (!check_probability(probability, "apply_classical_random_flip")) {
        return nullptr;
    }
    {
        AllowThreads allow;
        self->simulator->apply_classical_random_flip(static_cast<SoftCuda::Flt>(probability));
    }
    if (check_launch("apply_classical_random_flip") < 0) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* Simulator_apply_classical_check(PySimulator* self, PyObject* args) {
    if (require_simulator(self) < 0) {
        return nullptr;
    }
    SoftCuda::Err error = 0;
    if (!PyArg_ParseTuple(args, "i", &error)) {
        return nullptr;
    }
    {
        AllowThreads allow;
        self->simulator->apply_classical_check(error);
    }
    if (check_launch("apply_classical_check") < 0) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* Simulator_apply_classical_load_int(PySimulator* self, PyObject* args) {
    if (require_simulator(self) < 0) {
        return nullptr;
    }
    SoftCuda::Mid pointer = 0;
    if (!PyArg_ParseTuple(args, "I", &pointer)) {
        return nullptr;
    }
    {
        AllowThreads allow;
        self->simulator->apply_classical_load_int(pointer);
    }
    if (check_launch("apply_classical_load_int") < 0) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* Simulator_apply_classical_load_flt(PySimulator* self, PyObject* args) {
    if (require_simulator(self) < 0) {
        return nullptr;
    }
    SoftCuda::Mid pointer = 0;
    if (!PyArg_ParseTuple(args, "I", &pointer)) {
        return nullptr;
    }
    {
        AllowThreads allow;
        self->simulator->apply_classical_load_flt(pointer);
    }
    if (check_launch("apply_classical_load_flt") < 0) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* Simulator_apply_classical_save_int(PySimulator* self, PyObject* args) {
    if (require_simulator(self) < 0) {
        return nullptr;
    }
    SoftCuda::Mid pointer = 0;
    if (!PyArg_ParseTuple(args, "I", &pointer)) {
        return nullptr;
    }
    {
        AllowThreads allow;
        self->simulator->apply_classical_save_int(pointer);
    }
    if (check_launch("apply_classical_save_int") < 0) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* Simulator_apply_classical_save_flt(PySimulator* self, PyObject* args) {
    if (require_simulator(self) < 0) {
        return nullptr;
    }
    SoftCuda::Mid pointer = 0;
    if (!PyArg_ParseTuple(args, "I", &pointer)) {
        return nullptr;
    }
    {
        AllowThreads allow;
        self->simulator->apply_classical_save_flt(pointer);
    }
    if (check_launch("apply_classical_save_flt") < 0) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

using ReduceOp = void (SoftCuda::Simulator::*)(SoftCuda::ClassicalReduceArgs<>) const;

PyObject* call_reduce_op(PySimulator* self, PyObject* args, ReduceOp op, const char* name) {
    if (require_simulator(self) < 0) {
        return nullptr;
    }
    PyObject* pointers_object = nullptr;
    if (!PyArg_ParseTuple(args, "O", &pointers_object)) {
        return nullptr;
    }
    SoftCuda::ClassicalReduceArgs<> reduce_args{};
    if (!fill_reduce_args(pointers_object, reduce_args, self->args.mem_ints_n, name)) {
        return nullptr;
    }
    {
        AllowThreads allow;
        (self->simulator->*op)(reduce_args);
    }
    if (check_launch(name) < 0) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* Simulator_apply_classical_or(PySimulator* self, PyObject* args) {
    return call_reduce_op(
        self,
        args,
        &SoftCuda::Simulator::apply_classical_or,
        "apply_classical_or");
}

PyObject* Simulator_apply_classical_xor(PySimulator* self, PyObject* args) {
    return call_reduce_op(
        self,
        args,
        &SoftCuda::Simulator::apply_classical_xor,
        "apply_classical_xor");
}

PyObject* Simulator_apply_classical_and(PySimulator* self, PyObject* args) {
    return call_reduce_op(
        self,
        args,
        &SoftCuda::Simulator::apply_classical_and,
        "apply_classical_and");
}

PyObject* Simulator_apply_classical_match(PySimulator* self, PyObject* args) {
    if (require_simulator(self) < 0) {
        return nullptr;
    }
    PyObject* pointers_object = nullptr;
    PyObject* values_object = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &pointers_object, &values_object)) {
        return nullptr;
    }
    SoftCuda::ClassicalMatchArgs<> match_args{};
    if (!fill_match_args(pointers_object, values_object, match_args, self->args.mem_ints_n)) {
        return nullptr;
    }
    {
        AllowThreads allow;
        self->simulator->apply_classical_match(match_args);
    }
    if (check_launch("apply_classical_match") < 0) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* Simulator_apply_classical_controlled_x(PySimulator* self, PyObject* args) {
    return call_qid_op(
        self,
        args,
        &SoftCuda::Simulator::apply_classical_controlled_x,
        "apply_classical_controlled_x");
}

PyObject* Simulator_apply_classical_controlled_y(PySimulator* self, PyObject* args) {
    return call_qid_op(
        self,
        args,
        &SoftCuda::Simulator::apply_classical_controlled_y,
        "apply_classical_controlled_y");
}

PyObject* Simulator_apply_classical_controlled_z(PySimulator* self, PyObject* args) {
    return call_qid_op(
        self,
        args,
        &SoftCuda::Simulator::apply_classical_controlled_z,
        "apply_classical_controlled_z");
}

PyObject* Simulator_work_int(PySimulator* self, PyObject*) {
    return copy_per_shot<SoftCuda::Int>(
        self,
        NPY_INT,
        "work_int",
        [](SoftCuda::ShotStatePtr shot) { return shot.get_work_ptr().get_int_ptr(); });
}

PyObject* Simulator_work_float(PySimulator* self, PyObject*) {
    return copy_per_shot<SoftCuda::Flt>(
        self,
        NPY_DOUBLE,
        "work_float",
        [](SoftCuda::ShotStatePtr shot) { return shot.get_work_ptr().get_flt_ptr(); });
}

PyObject* Simulator_errors(PySimulator* self, PyObject*) {
    return copy_per_shot<SoftCuda::Err>(
        self,
        NPY_INT,
        "errors",
        [](SoftCuda::ShotStatePtr shot) { return shot.get_work_ptr().get_err_ptr(); });
}

PyObject* Simulator_entries_n(PySimulator* self, PyObject*) {
    return copy_per_shot<SoftCuda::Eid>(
        self,
        NPY_UINT,
        "entries_n",
        [](SoftCuda::ShotStatePtr shot) { return shot.get_entries_ptr().get_entries_n_ptr(); });
}

PyObject* Simulator_memory_ints(PySimulator* self, PyObject* args) {
    PyObject* pointer_object = Py_None;
    if (!PyArg_ParseTuple(args, "|O", &pointer_object)) {
        return nullptr;
    }
    if (pointer_object == Py_None) {
        return copy_memory_matrix<SoftCuda::Int>(
            self,
            self->args.mem_ints_n,
            NPY_INT,
            "memory_ints",
            [](SoftCuda::ShotStatePtr shot) { return shot.get_memory_ptr().get_ints_ptr(); });
    }

    SoftCuda::Mid pointer = 0;
    if (!parse_mid(pointer_object, pointer, "pointer")) {
        return nullptr;
    }
    if (pointer >= self->args.mem_ints_n) {
        PyErr_Format(
            PyExc_IndexError,
            "memory integer pointer %u is out of range for %u memory integers",
            pointer,
            self->args.mem_ints_n);
        return nullptr;
    }
    return copy_per_shot<SoftCuda::Int>(
        self,
        NPY_INT,
        "memory_ints",
        [pointer](SoftCuda::ShotStatePtr shot) {
            return shot.get_memory_ptr().get_int_ptr(pointer);
        });
}

PyObject* Simulator_memory_floats(PySimulator* self, PyObject* args) {
    PyObject* pointer_object = Py_None;
    if (!PyArg_ParseTuple(args, "|O", &pointer_object)) {
        return nullptr;
    }
    if (pointer_object == Py_None) {
        return copy_memory_matrix<SoftCuda::Flt>(
            self,
            self->args.mem_flts_n,
            NPY_DOUBLE,
            "memory_floats",
            [](SoftCuda::ShotStatePtr shot) { return shot.get_memory_ptr().get_flts_ptr(); });
    }

    SoftCuda::Mid pointer = 0;
    if (!parse_mid(pointer_object, pointer, "pointer")) {
        return nullptr;
    }
    if (pointer >= self->args.mem_flts_n) {
        PyErr_Format(
            PyExc_IndexError,
            "memory float pointer %u is out of range for %u memory floats",
            pointer,
            self->args.mem_flts_n);
        return nullptr;
    }
    return copy_per_shot<SoftCuda::Flt>(
        self,
        NPY_DOUBLE,
        "memory_floats",
        [pointer](SoftCuda::ShotStatePtr shot) {
            return shot.get_memory_ptr().get_flt_ptr(pointer);
        });
}

PyObject* Simulator_state_text(PySimulator* self, PyObject*) {
    if (require_simulator(self) < 0) {
        return nullptr;
    }

    const SoftCuda::ShotsStatePtr device_ptr = self->simulator->shots_state_ptr;
    const size_t bytes = device_ptr.get_size_bytes_n();
    std::vector<char> buffer;
    try {
        buffer.resize(bytes);
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    }

    cudaError_t error = cudaSuccess;
    {
        AllowThreads allow;
        error = cudaMemcpyAsync(
            buffer.data(),
            device_ptr.ptr,
            bytes,
            cudaMemcpyDeviceToHost,
            self->simulator->stream);
        if (error == cudaSuccess) {
            error = cudaStreamSynchronize(self->simulator->stream);
        }
    }
    if (set_cuda_error(error, "state_text") < 0) {
        return nullptr;
    }

    SoftCuda::ShotsStatePtr host_ptr = device_ptr;
    host_ptr.ptr = buffer.data();

    std::ostringstream out;
    write_shots_state(out, host_ptr);
    const std::string text = out.str();
    return PyUnicode_FromStringAndSize(text.data(), static_cast<Py_ssize_t>(text.size()));
}

PyObject* Simulator_get_args(PySimulator* self, void*) {
    return Py_BuildValue(
        "{s:I,s:I,s:I,s:I,s:I,s:I,s:d,s:K}",
        "shot_i",
        self->args.shot_i,
        "shots_n",
        self->args.shots_n,
        "qubits_n",
        self->args.qubits_n,
        "entries_m",
        self->args.entries_m,
        "mem_ints_n",
        self->args.mem_ints_n,
        "mem_flts_n",
        self->args.mem_flts_n,
        "epsilon",
        static_cast<double>(self->args.epsilon),
        "seed",
        self->args.seed);
}

PyObject* Simulator_get_shots_n(PySimulator* self, void*) {
    return PyLong_FromUnsignedLong(self->args.shots_n);
}

PyObject* Simulator_get_qubits_n(PySimulator* self, void*) {
    return PyLong_FromUnsignedLong(self->args.qubits_n);
}

PyObject* Simulator_get_entries_m(PySimulator* self, void*) {
    return PyLong_FromUnsignedLong(self->args.entries_m);
}

PyObject* Simulator_get_mem_ints_n(PySimulator* self, void*) {
    return PyLong_FromUnsignedLong(self->args.mem_ints_n);
}

PyObject* Simulator_get_mem_flts_n(PySimulator* self, void*) {
    return PyLong_FromUnsignedLong(self->args.mem_flts_n);
}

PyObject* Simulator_get_epsilon(PySimulator* self, void*) {
    return PyFloat_FromDouble(static_cast<double>(self->args.epsilon));
}

PyObject* cuda_runtime_version(PyObject*, PyObject*) {
    int version = 0;
    const cudaError_t error = cudaRuntimeGetVersion(&version);
    if (set_cuda_error(error, "cudaRuntimeGetVersion") < 0) {
        return nullptr;
    }
    return PyLong_FromLong(version);
}

PyObject* cuda_driver_version(PyObject*, PyObject*) {
    int version = 0;
    const cudaError_t error = cudaDriverGetVersion(&version);
    if (set_cuda_error(error, "cudaDriverGetVersion") < 0) {
        return nullptr;
    }
    return PyLong_FromLong(version);
}

PyObject* cuda_device_count(PyObject*, PyObject*) {
    int count = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (set_cuda_error(error, "cudaGetDeviceCount") < 0) {
        return nullptr;
    }
    return PyLong_FromLong(count);
}

PyMethodDef Simulator_methods[] = {
    {"close", reinterpret_cast<PyCFunction>(Simulator_close), METH_NOARGS, "Release CUDA resources."},
    {
        "synchronize",
        reinterpret_cast<PyCFunction>(Simulator_synchronize),
        METH_NOARGS,
        "Synchronize the simulator CUDA stream.",
    },
    {"apply_x", reinterpret_cast<PyCFunction>(Simulator_apply_x), METH_VARARGS, "Apply X."},
    {"apply_y", reinterpret_cast<PyCFunction>(Simulator_apply_y), METH_VARARGS, "Apply Y."},
    {"apply_z", reinterpret_cast<PyCFunction>(Simulator_apply_z), METH_VARARGS, "Apply Z."},
    {"apply_h", reinterpret_cast<PyCFunction>(Simulator_apply_h), METH_VARARGS, "Apply H."},
    {"apply_s", reinterpret_cast<PyCFunction>(Simulator_apply_s), METH_VARARGS, "Apply S."},
    {"apply_sdg", reinterpret_cast<PyCFunction>(Simulator_apply_sdg), METH_VARARGS, "Apply SDG."},
    {"apply_t", reinterpret_cast<PyCFunction>(Simulator_apply_t), METH_VARARGS, "Apply T."},
    {"apply_tdg", reinterpret_cast<PyCFunction>(Simulator_apply_tdg), METH_VARARGS, "Apply TDG."},
    {"apply_cx", reinterpret_cast<PyCFunction>(Simulator_apply_cx), METH_VARARGS, "Apply CX."},
    {
        "apply_measure",
        reinterpret_cast<PyCFunction>(Simulator_apply_measure),
        METH_VARARGS,
        "Measure a qubit.",
    },
    {
        "apply_desire",
        reinterpret_cast<PyCFunction>(Simulator_apply_desire),
        METH_VARARGS,
        "Measure a qubit with a desired result.",
    },
    {
        "apply_reset",
        reinterpret_cast<PyCFunction>(Simulator_apply_reset),
        METH_VARARGS,
        "Reset a qubit to a desired computational-basis value.",
    },
    {
        "apply_noise_x",
        reinterpret_cast<PyCFunction>(Simulator_apply_noise_x),
        METH_VARARGS,
        "Apply X noise.",
    },
    {
        "apply_noise_z",
        reinterpret_cast<PyCFunction>(Simulator_apply_noise_z),
        METH_VARARGS,
        "Apply Z noise.",
    },
    {
        "apply_noise_depo1",
        reinterpret_cast<PyCFunction>(Simulator_apply_noise_depo1),
        METH_VARARGS,
        "Apply single-qubit depolarizing noise.",
    },
    {
        "apply_noise_depo2",
        reinterpret_cast<PyCFunction>(Simulator_apply_noise_depo2),
        METH_VARARGS,
        "Apply two-qubit depolarizing noise.",
    },
    {
        "apply_classical_flip",
        reinterpret_cast<PyCFunction>(Simulator_apply_classical_flip),
        METH_NOARGS,
        "Flip the working integer as a boolean.",
    },
    {
        "apply_classical_random_flip",
        reinterpret_cast<PyCFunction>(Simulator_apply_classical_random_flip),
        METH_VARARGS,
        "Randomly flip the working integer as a boolean.",
    },
    {
        "apply_classical_check",
        reinterpret_cast<PyCFunction>(Simulator_apply_classical_check),
        METH_VARARGS,
        "Set an error code when the working integer is true.",
    },
    {
        "apply_classical_load_int",
        reinterpret_cast<PyCFunction>(Simulator_apply_classical_load_int),
        METH_VARARGS,
        "Load a memory integer into the working integer.",
    },
    {
        "apply_classical_load_flt",
        reinterpret_cast<PyCFunction>(Simulator_apply_classical_load_flt),
        METH_VARARGS,
        "Load a memory float into the working float.",
    },
    {
        "apply_classical_save_int",
        reinterpret_cast<PyCFunction>(Simulator_apply_classical_save_int),
        METH_VARARGS,
        "Save the working integer into memory.",
    },
    {
        "apply_classical_save_flt",
        reinterpret_cast<PyCFunction>(Simulator_apply_classical_save_flt),
        METH_VARARGS,
        "Save the working float into memory.",
    },
    {
        "apply_classical_or",
        reinterpret_cast<PyCFunction>(Simulator_apply_classical_or),
        METH_VARARGS,
        "Logical OR over memory integers.",
    },
    {
        "apply_classical_xor",
        reinterpret_cast<PyCFunction>(Simulator_apply_classical_xor),
        METH_VARARGS,
        "Logical XOR over memory integers.",
    },
    {
        "apply_classical_and",
        reinterpret_cast<PyCFunction>(Simulator_apply_classical_and),
        METH_VARARGS,
        "Logical AND over memory integers.",
    },
    {
        "apply_classical_match",
        reinterpret_cast<PyCFunction>(Simulator_apply_classical_match),
        METH_VARARGS,
        "Check memory integers against expected values.",
    },
    {
        "apply_classical_controlled_x",
        reinterpret_cast<PyCFunction>(Simulator_apply_classical_controlled_x),
        METH_VARARGS,
        "Classically controlled X.",
    },
    {
        "apply_classical_controlled_y",
        reinterpret_cast<PyCFunction>(Simulator_apply_classical_controlled_y),
        METH_VARARGS,
        "Classically controlled Y.",
    },
    {
        "apply_classical_controlled_z",
        reinterpret_cast<PyCFunction>(Simulator_apply_classical_controlled_z),
        METH_VARARGS,
        "Classically controlled Z.",
    },
    {"x", reinterpret_cast<PyCFunction>(Simulator_apply_x), METH_VARARGS, "Alias for apply_x."},
    {"y", reinterpret_cast<PyCFunction>(Simulator_apply_y), METH_VARARGS, "Alias for apply_y."},
    {"z", reinterpret_cast<PyCFunction>(Simulator_apply_z), METH_VARARGS, "Alias for apply_z."},
    {"h", reinterpret_cast<PyCFunction>(Simulator_apply_h), METH_VARARGS, "Alias for apply_h."},
    {"s", reinterpret_cast<PyCFunction>(Simulator_apply_s), METH_VARARGS, "Alias for apply_s."},
    {"sdg", reinterpret_cast<PyCFunction>(Simulator_apply_sdg), METH_VARARGS, "Alias for apply_sdg."},
    {"t", reinterpret_cast<PyCFunction>(Simulator_apply_t), METH_VARARGS, "Alias for apply_t."},
    {"tdg", reinterpret_cast<PyCFunction>(Simulator_apply_tdg), METH_VARARGS, "Alias for apply_tdg."},
    {"cx", reinterpret_cast<PyCFunction>(Simulator_apply_cx), METH_VARARGS, "Alias for apply_cx."},
    {
        "measure",
        reinterpret_cast<PyCFunction>(Simulator_apply_measure),
        METH_VARARGS,
        "Alias for apply_measure.",
    },
    {
        "desire",
        reinterpret_cast<PyCFunction>(Simulator_apply_desire),
        METH_VARARGS,
        "Alias for apply_desire.",
    },
    {
        "reset",
        reinterpret_cast<PyCFunction>(Simulator_apply_reset),
        METH_VARARGS,
        "Alias for apply_reset.",
    },
    {"work_int", reinterpret_cast<PyCFunction>(Simulator_work_int), METH_NOARGS, "Return working integers."},
    {
        "work_float",
        reinterpret_cast<PyCFunction>(Simulator_work_float),
        METH_NOARGS,
        "Return working floats.",
    },
    {"errors", reinterpret_cast<PyCFunction>(Simulator_errors), METH_NOARGS, "Return error codes."},
    {
        "entries_n",
        reinterpret_cast<PyCFunction>(Simulator_entries_n),
        METH_NOARGS,
        "Return current entry counts.",
    },
    {
        "memory_ints",
        reinterpret_cast<PyCFunction>(Simulator_memory_ints),
        METH_VARARGS,
        "Return all memory integers or one memory integer slot.",
    },
    {
        "memory_floats",
        reinterpret_cast<PyCFunction>(Simulator_memory_floats),
        METH_VARARGS,
        "Return all memory floats or one memory float slot.",
    },
    {
        "state_text",
        reinterpret_cast<PyCFunction>(Simulator_state_text),
        METH_NOARGS,
        "Return a textual state dump.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyGetSetDef Simulator_getset[] = {
    {"args", reinterpret_cast<getter>(Simulator_get_args), nullptr, "Simulator arguments.", nullptr},
    {"shots_n", reinterpret_cast<getter>(Simulator_get_shots_n), nullptr, "Number of shots.", nullptr},
    {"qubits_n", reinterpret_cast<getter>(Simulator_get_qubits_n), nullptr, "Number of qubits.", nullptr},
    {"entries_m", reinterpret_cast<getter>(Simulator_get_entries_m), nullptr, "Maximum entries.", nullptr},
    {
        "mem_ints_n",
        reinterpret_cast<getter>(Simulator_get_mem_ints_n),
        nullptr,
        "Number of memory integers.",
        nullptr,
    },
    {
        "mem_flts_n",
        reinterpret_cast<getter>(Simulator_get_mem_flts_n),
        nullptr,
        "Number of memory floats.",
        nullptr,
    },
    {"epsilon", reinterpret_cast<getter>(Simulator_get_epsilon), nullptr, "Pruning threshold.", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyObject* cuda_enabled(PyObject*, PyObject*) {
    Py_RETURN_TRUE;
}

PyMethodDef module_methods[] = {
    {
        "cuda_runtime_version",
        reinterpret_cast<PyCFunction>(cuda_runtime_version),
        METH_NOARGS,
        "Return the CUDA runtime version integer.",
    },
    {
        "cuda_driver_version",
        reinterpret_cast<PyCFunction>(cuda_driver_version),
        METH_NOARGS,
        "Return the CUDA driver version integer.",
    },
    {
        "cuda_device_count",
        reinterpret_cast<PyCFunction>(cuda_device_count),
        METH_NOARGS,
        "Return the number of CUDA devices visible to the runtime.",
    },
    {
        "cuda_enabled",
        reinterpret_cast<PyCFunction>(cuda_enabled),
        METH_NOARGS,
        "Return True when CUDA support is compiled into this extension.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "soft._native",
    "Native bindings for the SOFT CPU and CUDA simulators.",
    -1,
    module_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

int ready_simulator_type() {
    SimulatorType.tp_name = "soft._native.CudaSimulator";
    SimulatorType.tp_basicsize = sizeof(PySimulator);
    SimulatorType.tp_itemsize = 0;
    SimulatorType.tp_dealloc = reinterpret_cast<destructor>(Simulator_dealloc);
    SimulatorType.tp_flags = Py_TPFLAGS_DEFAULT;
    SimulatorType.tp_doc = "SOFT CUDA simulator.";
    SimulatorType.tp_methods = Simulator_methods;
    SimulatorType.tp_getset = Simulator_getset;
    SimulatorType.tp_init = reinterpret_cast<initproc>(Simulator_init);
    SimulatorType.tp_new = Simulator_new;
    return PyType_Ready(&SimulatorType);
}

}  // namespace

PyMODINIT_FUNC PyInit__native() {
    if (_import_array() < 0) {
        return nullptr;
    }

    SoftCudaError = PyErr_NewException("soft.SoftError", PyExc_RuntimeError, nullptr);
    if (SoftCudaError == nullptr) {
        return nullptr;
    }

    if (ready_cpu_circuit_type() < 0) {
        return nullptr;
    }

    if (ready_simulator_type() < 0) {
        return nullptr;
    }

    PyObject* module = PyModule_Create(&moduledef);
    if (module == nullptr) {
        return nullptr;
    }

    Py_INCREF(SoftCudaError);
    if (PyModule_AddObject(module, "SoftError", SoftCudaError) < 0) {
        Py_DECREF(SoftCudaError);
        Py_DECREF(module);
        return nullptr;
    }

    Py_INCREF(SoftCudaError);
    if (PyModule_AddObject(module, "SoftCudaError", SoftCudaError) < 0) {
        Py_DECREF(SoftCudaError);
        Py_DECREF(module);
        return nullptr;
    }

    Py_INCREF(&CpuCircuitType);
    if (PyModule_AddObject(module, "CpuCircuit", reinterpret_cast<PyObject*>(&CpuCircuitType)) < 0) {
        Py_DECREF(&CpuCircuitType);
        Py_DECREF(module);
        return nullptr;
    }

    Py_INCREF(&SimulatorType);
    if (PyModule_AddObject(module, "CudaSimulator", reinterpret_cast<PyObject*>(&SimulatorType)) < 0) {
        Py_DECREF(&SimulatorType);
        Py_DECREF(module);
        return nullptr;
    }

    Py_INCREF(&SimulatorType);
    if (PyModule_AddObject(module, "Simulator", reinterpret_cast<PyObject*>(&SimulatorType)) < 0) {
        Py_DECREF(&SimulatorType);
        Py_DECREF(module);
        return nullptr;
    }

    return module;
}

