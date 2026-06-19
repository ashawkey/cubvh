import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, IS_HIP_EXTENSION


# Fixed upstream in pytorch/pytorch#187665 (merged): BuildExtension now
# registers .hip with the MSVC compiler driver. This subclass is only needed
# on torch wheels built before that change.
class _BuildExt(BuildExtension):
    """Subclass that appends .hip to MSVC's _cpp_extensions on Windows.

    PyTorch's BuildExtension hipifies .cu -> .hip but does not register .hip
    with the MSVC compiler driver on Windows (only .cu/.cuh are added). Without
    this, MSVC's compile loop rejects .hip files before hipcc is ever invoked.
    The fix is a no-op on Linux where the HIP compiler is clang, not MSVC.
    """

    def build_extensions(self):
        if sys.platform == "win32" and hasattr(self.compiler, "_cpp_extensions"):
            self.compiler._cpp_extensions.append(".hip")
        super().build_extensions()

_src_path = os.path.dirname(os.path.abspath(__file__))

IS_WINDOWS = os.name == "nt"
IS_POSIX = os.name == "posix"

# ==========================================================
# Windows link workaround: c10.dll ValueError ctor not exported
# ==========================================================
# c10::ValueError inherits its constructors via "using Error::Error". When
# PyTorch is built with clang-cl (as TheRock does), clang-cl does not emit
# dllexport symbols for inherited constructors (llvm/llvm-project#162640),
# so ValueError's ctor is absent from c10.dll's export table.
# <torch/extension.h> pulls in headers that use TORCH_CHECK_VALUE (e.g.
# ATen/TensorIndexing.h), generating a __declspec(dllimport) reference that
# fails to link. Fixed upstream in pytorch/pytorch#175340 (merged): c10 now
# exports explicit constructors for the affected Error subclasses.
# Workaround: alias the missing thunk to Error(SourceLocation,string) which
# IS exported. Remove this block once a TheRock wheel ships a torch built
# with that fix.
extra_link_args = []
if IS_WINDOWS and IS_HIP_EXTENSION:
    _val_imp = (
        "__imp_??0ValueError@c10@@QEAA@USourceLocation@1@"
        "V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z"
    )
    _err_imp = (
        "__imp_??0Error@c10@@QEAA@USourceLocation@1@"
        "V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z"
    )
    extra_link_args.append(f"/ALTERNATENAME:{_val_imp}={_err_imp}")

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
# A ROCm torch build ships C++20 headers (c10/TensorImpl.h uses `requires`),
# so the HIP path needs C++20; the CUDA path stays at the upstream C++17.
cpp_standard = 20 if IS_HIP_EXTENSION else 17

# ==========================================================
# Device-compiler flags (NVCC for CUDA, hipcc for ROCm)
# ==========================================================
if IS_HIP_EXTENSION:
    # On a ROCm torch the .cu/.cuh sources are hipified and compiled by hipcc,
    # which does not accept the NVCC-only flags below. -DUSE_ROCM enables the
    # ROCm-specific shims in spcumc.cuh and api_gpu.cu. The offload arch is left
    # to torch's BuildExtension, which honors PYTORCH_ROCM_ARCH (or the native
    # GPUs when unset), so nothing is hardcoded here.
    base_nvcc_flags = [
        "-O3",
        f"-std=c++{cpp_standard}",
        "-DUSE_ROCM",
    ]
else:
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

    if not IS_HIP_EXTENSION:
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

    if not IS_HIP_EXTENSION:
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
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={
        "build_ext": _BuildExt,
    },
)
