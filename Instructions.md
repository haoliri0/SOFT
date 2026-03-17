# Instructions

## Build from source

### 1. Dependencies

To build from source, the following packages are needed:

* cmake
* gcc (or any other C++ compiler supported by cmake and nvcc)
* cuda toolkit (which includes nvcc and curand)

> The exact versions used during our development (on linux) are:
>  * cmake 3.22.1
>  * gcc 11.4.0
>  * cuda toolkit 12.1

> We currently only support linux or unix-like operating systems.
> Building on Windows is not supported yet.

### 2. Configuring

For convenience, we configure cmake using [presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html).
Specifically, we define cmake cache variables, environments, etc.
in file `CMakePresets.json` or `CMakeUserPresets.json` under the project root
to configure compilation arguments and paths to compilers, libraries, artifacts, etc.

However, neither `CMakePresets.json` nor `CMakeUserPresets.json` is committed to git,
as these configurations are usually environment-specific.
Instead, we provide an example presets file as `CMakeUserPresets.example.json`.
To quickly configure cmake, simply copy, rename, and modify this file.

> Using presets is not mandatory.
> If you are familiar with cmake,
> you can also configure it through command-line arguments and environment variables.

#### 3. Generating

Before compiling, cmake needs to generate a build system.
You can do this by running the following command:

```shell
cmake --preset Release
```

With this command, cmake generates build system files in a build folder `./cmake-build-release`.
The path to built folder is specified in the presets file.

Here `Release` is the name of a preset.
We provide 2 presets in the example presets file - `Release` and `Debug`,
which differ only in the value of `CMAKE_BUILD_TYPE`.
Preset `Release` tells the compiler to perform high-level optimizations.
Preset `Debug` tells the compiler to preserve information for debugging.

> If not using presets, you need to specify cache and environment variables manually.
> The command should be like:
> ```shell
> cmake \
>   -B <path-to-build-folder> \
>   -S <path-to-source-folder> \
>   -D <cache-variable-0>=<value-0> \
>   -D <cache-variable-1>=<value-1> \
>   <other-options> ...
> ```

#### 4. Compiling

With a generated build system, we can compile the source code with the following command:

```shell
cmake --build ./cmake-build-release
```

> If not using presets, you should run:
> ```shell
> cmake --build <path-to-build-folder>
> ```

After this command, you will find the built artifacts in the build folder `./cmake-build-release`:

* `cmake-build-release/soft-cuda-lib/libsoft-cuda.a`: a static library of the simulator.
* `cmake-build-release/soft-cuda-cli/soft-cuda-cli`: an executable file, the command-line interface of the simulator.

## Using the command-line interface

### 1. Overview

We provide a command-line interface (`soft-cuda-cli`),
which allows users to perform interactions via stdio.

The simulator maintains in memory (on GPU) the state of a system,
which consists of the quantum state of a fixed number of qubits,
a fixed number of classical integers and floats, running status, etc.
Users can interact with the system by applying a sequence of operations.

To leverage the power of GPUs, the simulator can run multiple independent shots in parallel.
Each shot shares the same sequence of operations
but has separate memory space and uses a different random seed.
The states of different shots are strictly isolated from each other in memory,
in other words, there is no interaction between shots.

The state of each shot consists of the following parts:

* `qubits`: The quantum state of a fixed number of qubits in generalized stabilizer representation.
* `working integer, working float`: An integer and a float for storing results, classical control, I/O, etc.
* `memory integers, memory floats`: Memory space to store a number of integers and floats.
* `error code`: An error code to indicate the running status of this shot.
    * If the error code is `0`, the simulation is successful.
    * If the error code is not `0`, all results of this shot will have undefined values.
    * Predefined error codes:
        * `0`: success - the simulation is successful.
        * `-1`: entries overflow error - the number of entries in the generalized stabilizer representation exceeds the limit.
        * `-2`: out of bounds error - the index of qubits, memory integers or memory floats is out of bounds.
* `hidden memory`: Other working space to ensure proper functioning, hidden from users.

### 2. Parameters

Here we introduce the command-line parameters of the simulator:

* `shots_n`: The number of shots. Default is `1`.
* `qubits_n`: The number of qubits. 
  * Limited by current implementation, only up to 64 qubits are supported.
  * We've planned to fix this in the next minor version.
* `mem_ints_n`: The number of memory integers.
* `mem_flts_n`: The number of memory floats.
* `entries_m`: The maximum number of entries in the map of generalized stabilizer representation.
    * Initially, there is only one entry in the map.
    * Some operations (such as T gate and measurement) may double the number of entries.
    * When the number of entries exceeds the limit, an entries overflow error will occur.
    * Smaller `entries_m` results in less memory consumption and better performance.
    * It is advised to try powers of two ($2^n$) and pick the smallest one that causes no overflow.
* `epsilon`: The threshold of modulus when pruning the map entries.
    * The simulator prunes the map every time it grows, removing the trivial entries.
    * When `epsilon>=0`, the simulator removes entries with modulus smaller than `epsilon`. This may result in a lossy approximation.
    * When `epsilon<0`, pruning is fully disabled. This may result in unnecessarily large map sizes due to numerical calculation errors.
    * The default value is `epsilon=0`, which guarantees a strict result.
    * In most experiments, we used `epsilon=1e-7` and observed no inconsistencies in outcomes.
* `seed`: A random seed (`uint64`) for all shots.
    * Running with the same seed and the same operations deterministically produces the same results.
    * Each shot uses a different random seed generated from this all-shots seed.
    * If not provided, the simulator will use a fixed default value (`0`)

Here is an example command to run the executable:

```shell
./cmake-build-release/soft-cuda-cli/soft-cuda-cli \
  --shots_n 1024 \
  --qubits_n 32 \
  --entries_m 2048 \
  --mem_ints_n 1024 \
  --mem_flts_n 1024 \
  --epsilon 1e-7 \
  --seed 42
```

Note that there is no command-line argument about the operations to perform (the quantum circuit to run).
By design, it should be passed through stdin instead.

### 3. Operations

Running the executable file in the command line will enter a shell-like environment,
where users can perform operations by entering corresponding texts.

The rules of this shell-like environment:

* Blank lines are ignored.
* One operation takes one line.
* An operation starts with its name, followed by arguments (split by spaces).
* Send `EOF` (type `Ctrl+D`) to exit.

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

* `MEASURE <target>`: measure specified qubit on computational basis.
* `DESIRE <target> <value>`: like `MEASURE`, but force the measurement result to be `value`.
* `RESET <target> <value>`: like `MEASURE`, but flip the target qubit if the measurement result is not the same as `value`.

After any of these measurement operations,
the measurement result is stored in the working integer (`0` or `1`),
and the probability of getting that result is stored in the working float.
You can save or print the measurement results by performing a `SAVE` or `PRINT` operation right after a measurement operation.

#### Quantum Noise Operations

* `XERR <prob> <target>`: apply a pauli X gate with probability `prob` on specified qubit.
* `YERR <prob> <target>`: apply a pauli Y gate with probability `prob` on specified qubit.
* `ZERR <prob> <target>`: apply a pauli Z gate with probability `prob` on specified qubit.
* `DEP1 <prob> <target>`: single-qubit depolarization, apply one of X, Y, Z gates with probability `prob`.
* `DEP2 <prob> <target0> <target1>`: two-qubit depolarization, apply one of XI, YI, ZI, XX, ..., ZZ (totally 15) gates with probability `prob`.

After any of these noise operations,
the type of sampled operation is stored in the working integer.
Correspondingly, the probability of getting this result is stored in the working float.
You can save or print the noise results by performing a `SAVE` or `PRINT` operation right after a noise operation.

> Taking `DEP1 <prob> <target>` as an example:
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
* `OR <index0> <index1> ...`: logical or on multiple memory integers (as boolean), storing result in working integer.
* `XOR <index0> <index1> ...`: logical xor on multiple memory integers (as boolean), storing result in working integer.
* `AND <index0> <index1> ...`: logical and on multiple memory integers (as boolean), storing result in working integer.
* `MATCH <index0> <index1> ... <value0> <value1> ...`: check whether all specified memory integers are equal to the corresponding given values, storing result in working integer.
* `CHECK <error>`: set the error code to `error` if working integer is not `0`.

#### I/O Operations

* `LOAD INT <index>`: load a memory integer into working integer.
* `LOAD FLT <index>`: load a memory float into working float.
* `SAVE INT <index>`: save working integer into memory integers.
* `SAVE FLT <index>`: save working float into memory floats.
* `PRINT INT`: print the working integer.
* `PRINT FLT`: print the working float.
* `PRINT ERR`: print the error code.
* `PRINT STATE`: print the detailed current state, for debugging only.
* `PRINT ENTRIES_N`: print the curren number of entries in the map of generalized stabilizer representation.

The `PRINT` operations print results via stdout in a YAML-compatible format.
This is the only way to read results from the simulator when using the command-line interface.
