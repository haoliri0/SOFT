from pathlib import Path
import os
import platform
import re
import shlex
import shutil
import subprocess

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist
from setuptools._distutils.errors import CompileError, DistutilsExecError


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_CPP_SRC = PACKAGE_ROOT.parent / "cpp" / "src"
CPP_SRC = REPOSITORY_CPP_SRC if REPOSITORY_CPP_SRC.exists() else PACKAGE_ROOT / "cpp" / "src"


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


ENABLE_CUDA = env_flag("SYMFT_PY_ENABLE_CUDA")
CUDA_REAL_DOUBLE = env_flag("SYMFT_PY_CUDA_REAL_DOUBLE")
CUDA_ARCH = os.environ.get("SYMFT_PY_CUDA_ARCH", "").strip()
CUDA_NVCC_FLAGS = os.environ.get("SYMFT_PY_CUDA_NVCC_FLAGS", "").strip()
ENABLE_NATIVE = env_flag("SYMFT_PY_NATIVE", default=True)


def find_cuda_home():
    for name in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(name)
        if value:
            return Path(value)
    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent.parent
    return None


def cuda_library_dirs(cuda_home):
    candidates = [cuda_home / "lib64", cuda_home / "lib", cuda_home / "lib" / "x64"]
    return [str(path) for path in candidates if path.exists()]


def cuda_nvcc(cuda_home):
    executable = "nvcc.exe" if platform.system() == "Windows" else "nvcc"
    candidate = cuda_home / "bin" / executable
    if candidate.exists():
        return str(candidate)
    found = shutil.which("nvcc")
    return found or str(candidate)



def _cuda_arch_digits(value):
    value = value.strip().lower()
    if not value:
        return None
    if value.startswith("sm_"):
        value = value[3:]
    elif value.startswith("compute_"):
        value = value[8:]
    value = value.replace(".", "")
    if value.isdigit():
        return value
    return None


def _cuda_arch_entries(value):
    return [entry for entry in re.split(r"[\s,;]+", value.strip()) if entry]


def cuda_arch_flags(value):
    flags = []
    for entry in _cuda_arch_entries(value):
        digits = _cuda_arch_digits(entry)
        if digits is None:
            flags.append(f"-arch={entry}")
        else:
            flags.append(f"-gencode=arch=compute_{digits},code=sm_{digits}")
    return flags


def detect_native_cuda_arches():
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []

    arches = []
    for line in result.stdout.splitlines():
        match = re.search(r"(\d+)\.(\d+)", line)
        if match:
            arches.append(f"{match.group(1)}{match.group(2)}")
    return sorted(set(arches))

CUDA_HOME = find_cuda_home() if ENABLE_CUDA else None


class BuildExt(build_ext):
    def finalize_options(self):
        super().finalize_options()
        import numpy

        self.include_dirs.append(numpy.get_include())

    def _compiler_supports(self, flags):
        if self.compiler.compiler_type != "unix":
            return False

        probe_dir = Path(self.build_temp) / "symft_flag_probes"
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_name = "_".join(flag.lstrip("-").replace("=", "_") for flag in flags)
        probe_source = probe_dir / f"{probe_name}.cpp"
        probe_source.write_text("int main() { return 0; }\n")
        objects = []
        try:
            objects = self.compiler.compile(
                [str(probe_source)],
                output_dir=str(probe_dir),
                extra_postargs=list(flags),
            )
        except (CompileError, DistutilsExecError, OSError):
            return False
        finally:
            probe_source.unlink(missing_ok=True)
            for obj in objects:
                Path(obj).unlink(missing_ok=True)
        return True

    def _configure_cpu_simd(self):
        source_flags = {}
        enabled = []
        machine = platform.machine().lower()
        if machine not in {"x86_64", "amd64", "i386", "i686", "x86"}:
            return source_flags, enabled

        backends = [
            (
                "avx2",
                ["-mavx2", "-mfma"],
                "simd/simd_avx2.cpp",
                "SYMFT_COMPILED_AVX2",
            ),
            (
                "avx512",
                ["-mavx512f", "-mavx512dq", "-mfma"],
                "simd/simd_avx512.cpp",
                "SYMFT_COMPILED_AVX512",
            ),
        ]
        for name, flags, relative_source, macro in backends:
            if not self._compiler_supports(flags):
                self.announce(
                    f"compiler does not support the SymFT {name} kernel flags; skipping it",
                    level=2,
                )
                continue
            source = cpp_source(relative_source)
            source_flags[Path(source).name] = flags
            for extension in self.extensions:
                if source not in extension.sources:
                    pending_optimizer = cpp_source("factored/pending_optimizer.cpp")
                    insert_at = (
                        extension.sources.index(pending_optimizer)
                        if pending_optimizer in extension.sources
                        else len(extension.sources)
                    )
                    extension.sources.insert(insert_at, source)
                macros = list(extension.define_macros or [])
                if not any(item[0] == macro for item in macros):
                    macros.append((macro, "1"))
                extension.define_macros = macros
            enabled.append(name)

        return source_flags, enabled

    def _configure_native_cpu(self):
        if not ENABLE_NATIVE or not self._compiler_supports(["-march=native"]):
            return False
        for extension in self.extensions:
            compile_args = list(extension.extra_compile_args or [])
            if "-march=native" not in compile_args:
                compile_args.append("-march=native")
            extension.extra_compile_args = compile_args

            macros = list(extension.define_macros or [])
            if not any(item[0] == "SYMFT_CPP_NATIVE_BUILD" for item in macros):
                macros.append(("SYMFT_CPP_NATIVE_BUILD", "1"))
            extension.define_macros = macros
        return True

    def build_extensions(self):
        native_cpu = self._configure_native_cpu()
        simd_source_flags, enabled_simd = self._configure_cpu_simd()
        effective_cuda_arch_flags = []
        effective_cuda_arch_state = CUDA_ARCH
        if ENABLE_CUDA:
            if CUDA_ARCH:
                effective_cuda_arch_flags = cuda_arch_flags(CUDA_ARCH)
            else:
                native_arches = detect_native_cuda_arches()
                if native_arches:
                    effective_cuda_arch_state = ",".join(f"sm_{arch}" for arch in native_arches)
                    effective_cuda_arch_flags = cuda_arch_flags(effective_cuda_arch_state)
                    self.announce(
                        "detected CUDA architecture(s): " + effective_cuda_arch_state,
                        level=3,
                    )
                else:
                    self.announce(
                        "could not detect CUDA architecture; set SYMFT_PY_CUDA_ARCH=sm_XX "
                        "to avoid relying on PTX JIT",
                        level=2,
                    )

        effective_cuda_arch_flags_state = " ".join(effective_cuda_arch_flags)
        state = (
            f"cuda={int(ENABLE_CUDA)};cuda_real_double={int(CUDA_REAL_DOUBLE)};"
            f"cuda_arch={effective_cuda_arch_state};"
            f"cuda_arch_flags={effective_cuda_arch_flags_state};"
            f"cuda_nvcc_flags={CUDA_NVCC_FLAGS};"
            f"cpu_native={int(native_cpu)};"
            f"cpu_simd={','.join(enabled_simd)}"
        )
        marker = Path(self.build_temp) / "symft_build_state.txt"
        if not marker.exists() or marker.read_text() != state:
            self.force = True

        nvcc = None
        if ENABLE_CUDA:
            cuda_home = CUDA_HOME or find_cuda_home()
            if cuda_home is None:
                raise RuntimeError(
                    "CUDA build requested with SYMFT_PY_ENABLE_CUDA=1, "
                    "but CUDA_HOME/CUDA_PATH or nvcc was not found"
                )
            nvcc = cuda_nvcc(cuda_home)
            if ".cu" not in self.compiler.src_extensions:
                self.compiler.src_extensions.append(".cu")

        original_compile = self.compiler._compile

        def compile_with_backends(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith(".cu"):
                nvcc_args = [nvcc, "-c", src, "-o", obj, *cc_args]
                if platform.system() == "Windows":
                    nvcc_args.extend(["-std=c++20", "-O2"])
                else:
                    nvcc_args.extend(["-std=c++20", "-O3", "--compiler-options", "-fPIC"])
                nvcc_args.extend(effective_cuda_arch_flags)
                if CUDA_NVCC_FLAGS:
                    nvcc_args.extend(shlex.split(CUDA_NVCC_FLAGS))
                self.spawn(nvcc_args)
                return

            source_args = list(extra_postargs or [])
            source_args.extend(simd_source_flags.get(Path(src).name, []))
            original_compile(obj, src, ext, cc_args, source_args, pp_opts)

        self.compiler._compile = compile_with_backends

        super().build_extensions()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(state)


class Sdist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        shutil.copytree(CPP_SRC, Path(base_dir) / "cpp" / "src")


def cpp_source(path):
    return str(CPP_SRC / path)


sources = [
    "src/symft/_native.cpp",
    cpp_source("core/common.cpp"),
    cpp_source("core/pauli.cpp"),
    cpp_source("core/symbolic.cpp"),
    cpp_source("core/frames.cpp"),
    cpp_source("circuit/circuit_lowering.cpp"),
    cpp_source("sampler/active_state.cpp"),
    cpp_source("sampler/contiguous_active.cpp"),
    cpp_source("sampler/component_plan.cpp"),
    cpp_source("factored/factored_state.cpp"),
    cpp_source("factored/factored_planner.cpp"),
    cpp_source("sampler/presampled_expression.cpp"),
    cpp_source("sampler/single_shot_sampler.cpp"),
    cpp_source("sampler/prepared_sampler.cpp"),
    cpp_source("sampler/exogenous_presample.cpp"),
    cpp_source("frontend/stim_parser.cpp"),
    cpp_source("frontend/stim_sampling.cpp"),
    cpp_source("sampler/batch_runtime.cpp"),
    cpp_source("sampler/batch_symbols.cpp"),
    cpp_source("sampler/batch_active.cpp"),
    cpp_source("simd/simd_dispatch.cpp"),
    cpp_source("simd/simd_scalar.cpp"),
    cpp_source("simd/batch_interleaved.cpp"),
    cpp_source("factored/pending_optimizer.cpp"),
]

if ENABLE_CUDA:
    sources.extend(
        [
            cpp_source("cuda/cuda_program.cpp"),
            cpp_source("cuda/cuda_sampler.cpp"),
            cpp_source("cuda/cuda_runtime.cu"),
        ]
    )

if platform.system() == "Windows":
    compile_args = ["/std:c++20", "/O2"]
else:
    compile_args = ["-std=c++20", "-O3", "-fvisibility=hidden"]
link_args = []
if platform.system() == "Linux":
    compile_args.append("-pthread")
    link_args.append("-pthread")

include_dirs = [str(CPP_SRC)]
library_dirs = []
runtime_library_dirs = []
libraries = []
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

if ENABLE_CUDA:
    define_macros.append(("SYMFT_CPP_ENABLE_CUDA", "1"))
    if CUDA_REAL_DOUBLE:
        define_macros.append(("SYMFT_CUDA_REAL_DOUBLE", "1"))
    libraries.append("cudart")
    if CUDA_HOME is not None:
        include_dirs.append(str(CUDA_HOME / "include"))
        library_dirs.extend(cuda_library_dirs(CUDA_HOME))
        if platform.system() == "Linux":
            runtime_library_dirs.extend(cuda_library_dirs(CUDA_HOME))

extensions = [
    Extension(
        "symft._native",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        runtime_library_dirs=runtime_library_dirs,
        libraries=libraries,
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=define_macros,
    )
]

setup(
    name="symft",
    version="0.1.0",
    description="Python bindings for the SymFT Clifford+T simulator",
    license="Apache-2.0",
    license_files=["LICENSE"],
    classifiers=["License :: OSI Approved :: Apache Software License"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"symft": ["py.typed", "*.pyi"]},
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExt, "sdist": Sdist},
    install_requires=["numpy>=1.20"],
    python_requires=">=3.9",
    zip_safe=False,
)
