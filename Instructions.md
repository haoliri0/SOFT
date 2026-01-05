# Instructions

## Build from source

### 1. Dependencies

To build from source, the following packages are needed:

* cmake
* gcc (or any other C++ compiler supported by cmake and nvcc)
* cuda toolkit (includes nvcc compiler and curand library)

> The exact versions used during our development (on linux) are listed below:
>  * cmake 3.22.1
>  * gcc 11.4.0
>  * cuda toolkit 12.1

> To build on Windows, you may need to install [Visual Studio](https://visualstudio.microsoft.com/),
> since cuda toolkit supports only MSVC compiler on Windows.

### 2. Configuring

For convenience, we configure cmake using [presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html).
Specifically, we define the cmake cache variables, environments, etc.
in file `CMakePresets.json` or `CMakeUserPresets.json` under the project root
to configure arguments for compilation, path to the compiler, libraries, artifacts, etc.

However, none of `CMakePresets.json` or `CMakeUserPresets.json` is commited in git,
because it usually works only in local environment.
Instead, we provide an example of presets file as `CMakeUserPresets.example.json`.
To quickly configure cmake, you can simply copy, rename and modify this file.

> Using presets is not mandatory.
> If you are familiar with cmake,
> you may also configure it through command-line arguments and environment variables.

#### 3. Generating

Before compiling, cmake need to generate a build system.
You can do this by running the following command.

```shell
cmake --preset Release
```

With this command, cmake generates build system files in a build folder `./cmake-build-release`,
which is specified in presets file.
Here `Release` is the name of a preset.
We provide 2 presets in the example presets file - `Release` and `Debug`.
They only differ in the value of `CMAKE_BUILD_TYPE`.
`Release` tells the compiler to perform high-level optimizations,
while `Debug` tells the compiler to preserve debug information.

> If not using presets, the command should be like:
> ```shell
> cmake -S <path-to-source-folder> -B <path-to-build-folder>
> ```
>
> And you may specify cache and environment variables by yourself.

#### 4. Compiling

Then we can compile the source code by running the following command.

```shell
cmake --build ./cmake-build-release
```

> If not using presets, you should run:
> ```shell
> cmake --build <path-to-build-folder>
> ```

After this command, you will find built artifacts in the build folder `./cmake-build-release`.

* `libsoft_cuda.a`: a static library of the simulator.
* `soft_cuda_exec`: an executable file, a command-line interface of the simulator.

## Using the command-line interface

### 1. Overview

We provide a command-line interface (the executable file `soft_cuda_exec`),
which allows us to perform the interactions via stdio.

The simulator maintains in memory (on GPU) the state of a system,
which consists of a quantum state of a fixed number of qubits,
a fixed number of classical integers and floats, running status, etc.
User can interact with the system by applying a sequence of operations.

To leverage the power of GPU, the simulator can run multiple independent shots in parallel.
Each shot shares the same sequence of operation, 
but owns separate memory space and uses a different random seed.
The states of different shots are strictly isolated from each other in memory,
in other words, there is no interaction between shots.

### 2. State

The state of each shot roughly consists of the following parts.

* `qubits`: The quantum state of fixed number of qubits in generalized stabilizer representation.
* `working integer, working float`: A integer and a float for storing results, classical control, io, etc.
* `memory integers, memory floats`: A memory space to store a number of integers and floats.
* `error code`: An error code to indicate the running status of this shot.
    * If the error code is `0`, the simulation is successful.
    * If the error code is not `0`, all results of this shot will have undefined values.
* `hidden memory`: Other working space to ensure proper functioning, hidden from users.

### 3. Parameters

Here we introduce the command-line parameters of the simulator.

* `shots_n`: The number of shots. Default `1`.
* `qubits_n`: The number of qubits.
* `mem_ints_m`: The number of memory integers.
* `mem_flts_m`: The number of memory floats.
* `entries_m`: The maximum number of entries in the map of generalized stabilizer representation.
    * Initially, there is only one entry in the map.
    * Some operations (such as T gate and measurement) may double the number of entries.
    * When the number of entries exceeds the limit, an entries overflow error will occur.
    * Adjust it to the smallest value that causes few enough errors.
    * Try $2^n$ values as the number of entries always grows in multiples of $2$.
* `epsilon`: The threshold of modulus when pruning the map entries.
    * The simulator prunes the map everytime it grows to remove the entries which are closed to zero.
    * When `epsilon>=0`, the simulator removes entries which have a smaller modulus than `epsilon`. This may result in a lossy approximation.
    * When `epsilon<0`, the pruning is fully disabled. It may result in unnecessarily larger map sizes due to numerical calculation errors.
    * The default value is `epsilon=0`, which guarantees a strict result.
    * In most experiments we used `epsilon=1e-7` and observed no inconsistencies in outcomings.
* `seed`: A random seed (`uint64`) for all shots.
    * Running with the same seed and the same operations produces deterministic same results.
    * Each shot uses a different random seed generated from this all-shots seed.
    * If not provided, the simulator will use a fixed default value (`0`)

Here is an example command to run the executable:

```shell
./cmake-build-release/soft_cuda_exec \
  --shots_n 1024 \
  --qubits_n 32 \
  --entries_m 2048 \
  --mem_ints_m 1024 \
  --mem_flts_m 1024 \
  --epsilon 1e-7 \
  --seed 42
```

Note that the operations to perform (the quantum circuit to run) 
are passed through stdin, rather than through command-line arguments.

### 4. Operations

Running an executable file in the command line will enter a shell-like environment,
where user can perform operations by entering corresponding texts.

Rules are simple:

* Blank lines are ignored;
* One operation takes one line;
* An operation starts with its name, following by arguments (split by spaces);
* Send `EOF` (type `Ctrl+D`) to exit;

All allowed operations are listed below.

#### Gates

* `X <target>`: perform pauli X gate on specified qubit.
* `Y <target>`: perform pauli Y gate on specified qubit.
* `Z <target>`: perform pauli Z gate on specified qubit.
* `H <target>`: perform Hadamard gate on specified qubit.
* `S <target>`: perform S gate on specified qubit.
* `SDG <target>`: perform inverted S gate on specified qubit.
* `T <target>`: perform T gate on specified qubit.
* `TDG <target>`: perform inverted T gate on specified qubit.
* `CX <control> <target>`: perform CNOT gate on specified qubits.

#### Measurements

* `MEASURE <target>`: measure specified qubit on computational basics.
* `DESIRE <target> <value>`: like `MEASURE`, but force the measurement result to be `value`.
* `RESET <target> <value>`: like `MEASURE`, but flip the target qubit if the measurement result is not the same as`value`.

After any one of these measurement operations,
the measurement result is stored in the working integer (`0` or `1`),
and the probability of getting that result is stored in the working float.
You can save or print the measurement results by performing a `SAVE` or `PRINT` operation right after a measurement operation.

#### Quantum Noise Operations

* `XERR <prob> <target>`: apply a pauli X gate with probability `prob` on specified qubit.
* `YERR <prob> <target>`: apply a pauli Y gate with probability `prob` on specified qubit.
* `ZERR <prob> <target>`: apply a pauli Z gate with probability `prob` on specified qubit.
* `DEP1 <prob> <target>`: single-qubit depolarization, apply one of X, Y, Z gate with probability `prob/3`.
* `DEP2 <prob> <target0> <target1>`: two-qubit depolarization, apply one of XI, YI, ZI, XX, ..., ZZ gate with probability `prob/15`.

After any one of these noise operations,
the type of sampled operation is stored in the working integer.
Correspondingly, the probability of getting such result is stored in the working float.
You can save or print the noise results by performing a `SAVE` or `PRINT` operation right after a noise operation.

> Take `DEP1 <prob> <target>` as an example:
> * if X gate is applied, the working integer will be `1`, the working float will be `prob/3`;
> * if Y gate is applied, the working integer will be `2`, the working float will be `prob/3`;
> * if Z gate is applied, the working integer will be `3`, the working float will be `prob/3`;
> * if no gate is applied, the working integer will be `0`, the working float will be `1-prob`.

#### Classical Controlled Gates

* `CCX <target>`: apply pauli X gate to specified qubit if working integer is not `0`.
* `CCY <target>`: apply pauli Y gate to specified qubit if working integer is not `0`.
* `CCZ <target>`: apply pauli Z gate to specified qubit if working integer is not `0`.

#### Boolean Operations

* `FLIP`: flip the value of working integer as a boolean.
* `RANDFLIP <prob>`: perform `FLIP` with probability `prob`.
* `OR <index0> <index1> ...`: logical or on multiple memory integers (as boolean).
    * Result is stored in working integer.
* `XOR <index0> <index1> ...`: logical xor on multiple memory integers (as boolean).
    * Result is stored in working integer.
* `AND <index0> <index1> ...`: logical and on multiple memory integers (as boolean).
    * Result is stored in working integer.
* `MATCH <index0> <index1> ... <value0> <value1> ...`: check whether the specified memory integers are equal to the specified values.
    * Result is stored in working integer.
* `CHECK <error>`: set the error code to `error` if working integer is not `0`.

#### IO Operations

* `LOAD INT <index>`: load a memory integer into working integer.
* `LOAD FLT <index>`: load a memory float into working float.
* `SAVE INT <index>`: save working integer into memory integers.
* `SAVE FLT <index>`: save working float into memory floats.
* `PRINT INT`: print the working integer.
* `PRINT FLT`: print the working float.
* `PRINT ERR`: print the error code.
* `PRINT STATE`: print the detailed current state, only for debugging.

The `PRINT` operations print the result via stdout in a `yaml` compatible format.
This is the only way to read results from the simulator when using the command-line interface.
