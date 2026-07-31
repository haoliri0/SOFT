# Instructions for Build from source

## 1. Dependencies

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

## 2. Configuring

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

## 3. Generating

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

## 4. Compiling

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
