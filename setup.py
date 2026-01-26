import os
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

    # The following definitions must be undefined
    # since we need half-precision operation.
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
        # `-allow-unsupported-compiler` suppresses NVCC host-compiler version checks
		#  so that modern MSVC (VS2022+ builds) can compile with recent CUDA (12/13).
		#  This flag does NOT disable correctness safety checks.
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
'''
Usage:
python setup.py build_ext --inplace # build extensions locally, do not install (only can be used from the parent directory)
python setup.py install # build extensions and install (copy) to PATH.
pip install . # ditto but better (e.g., dependency & metadata handling)
python setup.py develop # build extensions and install (symbolic) to PATH.
pip install -e . # ditto but better (e.g., dependency & metadata handling)
'''
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
