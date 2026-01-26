import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

IS_WINDOWS = os.name == "nt"
IS_POSIX = os.name == "posix"

# ==========================================================
# Windows: ensure MSVC environment is available
# ==========================================================
if IS_WINDOWS:
    def find_cl_path():
        import glob
        for executable in ["Program Files (x86)", "Program Files"]:
            for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
                paths = sorted(
                    glob.glob(
                        f"C:\\{executable}\\Microsoft Visual Studio\\*\\{edition}\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64"
                    ),
                    reverse=True,
                )
                if paths:
                    return paths[0]

    # If cl.exe not found, try to locate it
    if os.system("where cl.exe >nul 2>nul") != 0:
        cl_path = find_cl_path()
        if cl_path is None:
            raise RuntimeError("Could not locate a supported MSVC installation")
        os.environ["PATH"] += ";" + cl_path
    else:
        # Already in dev prompt
        os.environ["DISTUTILS_USE_SDK"] = "1"

# ==========================================================
# Common config
# ==========================================================
cpp_standard = 17

# ==========================================================
# NVCC flags (shared)
# ==========================================================
base_nvcc_flags = [
    "-O3",
    f"-std=c++{cpp_standard}",
    "--extended-lambda",
    "--expt-relaxed-constexpr",

    # Required for half precision
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

# ==========================================================
# Platform-specific flags
# ==========================================================
if IS_POSIX:
    base_cflags = [
        "-O3",
        f"-std=c++{cpp_standard}",
    ]

    base_nvcc_flags += [
        "-Xcompiler=-Wno-float-conversion",
        "-Xcompiler=-fno-strict-aliasing",
    ]

elif IS_WINDOWS:
    base_cflags = [
        "/O2",
        f"/std:c++{cpp_standard}",

        # CRITICAL: modern MSVC conformance
        "/permissive-",
        "/Zc:__cplusplus",

        # Required for PyTorch / pybind11 exception safety
        "/EHsc",
    ]

    base_nvcc_flags += [
        "-allow-unsupported-compiler",

        # Propagate host flags into NVCC host compiler
        f"-Xcompiler=/std:c++{cpp_standard}",
        "-Xcompiler=/permissive-",
        "-Xcompiler=/Zc:__cplusplus",
        "-Xcompiler=/EHsc",
    ]

# ==========================================================
# Extension
# ==========================================================
setup(
    ext_modules=[
        CUDAExtension(
            name="_cubvh",
            sources=[
                os.path.join("src", "bvh.cu"),
                os.path.join("src", "api_gpu.cu"),
                os.path.join("src", "bindings.cpp"),
            ],
            include_dirs=[
                os.path.join(_src_path, "include"),
                os.path.join(_src_path, "third_party", "eigen"),
            ],
            extra_compile_args={
                "cxx": base_cflags,
                "nvcc": base_nvcc_flags,
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
