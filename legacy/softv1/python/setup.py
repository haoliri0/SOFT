from pathlib import Path
import os
import platform
import shlex
import shutil

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parent
REPOSITORY_SOFT_LIB = REPOSITORY_ROOT / "soft-cuda-lib"
REPOSITORY_SOFT_CPU = REPOSITORY_ROOT / "soft-cpu-lib"
REPOSITORY_SOFT_CLI = REPOSITORY_ROOT / "soft-cuda-cli"

SOFT_LIB = REPOSITORY_SOFT_LIB if REPOSITORY_SOFT_LIB.exists() else PACKAGE_ROOT / "soft-cuda-lib"
SOFT_CPU = REPOSITORY_SOFT_CPU if REPOSITORY_SOFT_CPU.exists() else PACKAGE_ROOT / "soft-cpu-lib"
SOFT_CLI = REPOSITORY_SOFT_CLI if REPOSITORY_SOFT_CLI.exists() else PACKAGE_ROOT / "soft-cuda-cli"

CUDA_ARCH = os.environ.get("SOFT_PY_CUDA_ARCH", "native").strip()
CUDA_NVCC_FLAGS = os.environ.get("SOFT_PY_NVCC_FLAGS", "").strip()


def find_cuda_home():
    for name in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(name)
        if value:
            return Path(value)
    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent.parent
    for candidate in (
        Path("/usr/local/cuda"),
        Path("/usr/local/cuda-13.2"),
        Path("/usr/local/cuda-13"),
        Path("/usr/local/cuda-12.8"),
        Path("/usr/local/cuda-12"),
    ):
        if (candidate / "bin" / "nvcc").exists():
            return candidate
    return None


def cuda_library_dirs(cuda_home):
    candidates = [
        cuda_home / "lib64",
        cuda_home / "lib",
        cuda_home / "lib" / "x64",
        cuda_home / "targets" / "x86_64-linux" / "lib",
    ]
    return [str(path) for path in candidates if path.exists()]


def cuda_nvcc(cuda_home):
    executable = "nvcc.exe" if platform.system() == "Windows" else "nvcc"
    candidate = cuda_home / "bin" / executable
    if candidate.exists():
        return str(candidate)
    found = shutil.which("nvcc")
    return found or str(candidate)


CUDA_HOME = find_cuda_home()


class BuildExt(build_ext):
    def finalize_options(self):
        super().finalize_options()
        import numpy

        self.include_dirs.append(numpy.get_include())

    def build_extensions(self):
        cuda_home = CUDA_HOME or find_cuda_home()
        if cuda_home is None:
            raise RuntimeError(
                "CUDA toolkit is required to build soft. "
                "Set CUDA_HOME/CUDA_PATH or make nvcc available on PATH."
            )

        nvcc = cuda_nvcc(cuda_home)
        if ".cu" not in self.compiler.src_extensions:
            self.compiler.src_extensions.append(".cu")

        original_compile = self.compiler._compile

        def compile_with_nvcc(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith(".cu"):
                nvcc_args = [nvcc, "-c", src, "-o", obj, *cc_args]
                if platform.system() == "Windows":
                    nvcc_args.extend(["-std=c++20", "-O2"])
                else:
                    nvcc_args.extend(["-std=c++20", "-O3", "--compiler-options", "-fPIC"])
                if CUDA_ARCH:
                    nvcc_args.append(f"-arch={CUDA_ARCH}")
                if CUDA_NVCC_FLAGS:
                    nvcc_args.extend(shlex.split(CUDA_NVCC_FLAGS))
                self.spawn(nvcc_args)
            else:
                original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = compile_with_nvcc
        super().build_extensions()


class Sdist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        shutil.copytree(SOFT_LIB, Path(base_dir) / "soft-cuda-lib")
        shutil.copytree(SOFT_CPU, Path(base_dir) / "soft-cpu-lib")
        shutil.copytree(SOFT_CLI, Path(base_dir) / "soft-cuda-cli")


def lib_source(path):
    return str(SOFT_LIB / "source" / path)

def cpu_source(path):
    return str(SOFT_CPU / "src" / path)


sources = [
    "src/soft/_native.cpp",
    cpu_source("stabilizers.cpp"),
    lib_source("simulator.cu"),
    lib_source("decompose.cu"),
    lib_source("classical.cu"),
    lib_source("measure.cu"),
    lib_source("noises.cu"),
    lib_source("gates_t.cu"),
    lib_source("gates.cu"),
]

if platform.system() == "Windows":
    compile_args = ["/std:c++20", "/O2"]
else:
    compile_args = ["-std=c++20", "-O3", "-fvisibility=hidden"]

link_args = []
if platform.system() == "Linux":
    compile_args.append("-pthread")
    link_args.append("-pthread")

include_dirs = [
    str(SOFT_LIB / "include"),
    str(SOFT_LIB / "source"),
    str(SOFT_CPU / "include"),
    str(SOFT_CPU / "src"),
    str(SOFT_CLI / "source"),
]
library_dirs = []
runtime_library_dirs = []
libraries = ["cudart"]

if CUDA_HOME is not None:
    include_dirs.append(str(CUDA_HOME / "include"))
    library_dirs.extend(cuda_library_dirs(CUDA_HOME))
    if platform.system() == "Linux":
        runtime_library_dirs.extend(cuda_library_dirs(CUDA_HOME))

extensions = [
    Extension(
        "soft._native",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs=runtime_library_dirs,
        libraries=libraries,
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]


setup(
    name="soft",
    version="0.1.0",
    description="Python bindings for the SOFT CPU and CUDA simulators",
    packages=["soft"],
    package_dir={"soft": "src/soft"},
    package_data={"soft": ["py.typed", "*.pyi"]},
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExt, "sdist": Sdist},
    install_requires=["numpy>=1.20"],
    python_requires=">=3.9",
    zip_safe=False,
)

