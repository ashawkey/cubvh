import os
import re
import subprocess
from pkg_resources import parse_version
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))

# ref: https://github.com/sxyu/sdf/blob/master/setup.py
def find_eigen(min_ver=(3, 3, 0)):
	"""Helper to find or download the Eigen C++ library"""
	import re, os
	try_paths = [
		'/usr/include/eigen3',
		'/usr/local/include/eigen3',
		os.path.expanduser('~/.local/include/eigen3'),
		'C:/Program Files/eigen3',
		'C:/Program Files (x86)/eigen3',
	]
	WORLD_VER_STR = "#define EIGEN_WORLD_VERSION"
	MAJOR_VER_STR = "#define EIGEN_MAJOR_VERSION"
	MINOR_VER_STR = "#define EIGEN_MINOR_VERSION"
	EIGEN_WEB_URL = 'https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2'
	TMP_EIGEN_FILE = 'tmp_eigen.tar.bz2'
	TMP_EIGEN_DIR = './eigen3.3.7'
	min_ver_str = '.'.join(map(str, min_ver))

	eigen_path = None
	for path in try_paths:
		macros_path = os.path.join(path, 'Eigen/src/Core/util/Macros.h')
		if os.path.exists(macros_path):
			macros = open(macros_path, 'r').read().split('\n')
			world_ver, major_ver, minor_ver = None, None, None
			for line in macros:
				if line.startswith(WORLD_VER_STR):
					world_ver = int(line[len(WORLD_VER_STR):])
				elif line.startswith(MAJOR_VER_STR):
					major_ver = int(line[len(MAJOR_VER_STR):])
				elif line.startswith(MINOR_VER_STR):
					minor_ver = int(line[len(MINOR_VER_STR):])
			if world_ver is None or major_ver is None or minor_ver is None:
				print('Failed to parse macros file', macros_path)
			else:
				ver = (world_ver, major_ver, minor_ver)
				ver_str = '.'.join(map(str, ver))
				if ver < min_ver:
					print('Found unsuitable Eigen version', ver_str, 'at',
						  path, '(need >= ' + min_ver_str + ')')
				else:
					print('Found Eigen version', ver_str, 'at', path)
					eigen_path = path
					break

	if eigen_path is None:
		try:
			import urllib.request
			print("Couldn't find Eigen locally, downloading...")
			req = urllib.request.Request(
				EIGEN_WEB_URL,
				data=None,
				headers={
					'User-Agent':
					'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'
				})

			with urllib.request.urlopen(req) as resp,\
				  open(TMP_EIGEN_FILE, 'wb') as file:
				data = resp.read()
				file.write(data)
			import tarfile
			tar = tarfile.open(TMP_EIGEN_FILE)
			tar.extractall()
			tar.close()

			eigen_path = TMP_EIGEN_DIR
			os.remove(TMP_EIGEN_FILE)
		except:
			print('Download failed, failed to find Eigen')

	if eigen_path is not None:
		print('Found eigen at', eigen_path)

	return eigen_path


if os.name == "nt":

	# find cl.exe
	def find_cl_path():
		import glob
		for executable in ["Program Files (x86)", "Program Files"]:
			for edition in ["Enterprise", "Professional", "BuildTools", "Community"]:
				paths = sorted(glob.glob(f"C:\\{executable}\\Microsoft Visual Studio\\*\\{edition}\\VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64"), reverse=True)
				if paths:
					return paths[0]

	# If cl.exe is not on path, try to find it.
	if os.system("where cl.exe >nul 2>nul") != 0:
		cl_path = find_cl_path()
		if cl_path is None:
			raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
		os.environ["PATH"] += ";" + cl_path
	else:
		# cl.exe was found in PATH, so we can assume that the user is already in a developer command prompt
		# In this case, BuildExtensions requires the following environment variable to be set such that it
		# won't try to activate a developer command prompt a second time.
		os.environ["DISTUTILS_USE_SDK"] = "1"

cpp_standard = 14

# Get CUDA version and make sure the targeted compute capability is compatible
if os.system("nvcc --version") == 0:
	nvcc_out = subprocess.check_output(["nvcc", "--version"]).decode()
	cuda_version = re.search(r"release (\S+),", nvcc_out)

	if cuda_version:
		cuda_version = parse_version(cuda_version.group(1))
		print(f"Detected CUDA version {cuda_version}")
		if cuda_version >= parse_version("11.0"):
			cpp_standard = 17

print(f"Targeting C++ standard {cpp_standard}")


base_nvcc_flags = [
	f"-std=c++{cpp_standard}",
	"--extended-lambda",
	"--expt-relaxed-constexpr",
	# The following definitions must be undefined
	# since TCNN requires half-precision operation.
	"-U__CUDA_NO_HALF_OPERATORS__",
	"-U__CUDA_NO_HALF_CONVERSIONS__",
	"-U__CUDA_NO_HALF2_OPERATORS__",
]

if os.name == "posix":
	base_cflags = [f"-std=c++{cpp_standard}"]
	base_nvcc_flags += [
		"-Xcompiler=-Wno-float-conversion",
		"-Xcompiler=-fno-strict-aliasing",
	]
elif os.name == "nt":
	base_cflags = [f"/std:c++{cpp_standard}"]

'''
Usage:
python setup.py build_ext --inplace # build extensions locally, do not install (only can be used from the parent directory)
python setup.py install # build extensions and install (copy) to PATH.
pip install . # ditto but better (e.g., dependency & metadata handling)
python setup.py develop # build extensions and install (symbolic) to PATH.
pip install -e . # ditto but better (e.g., dependency & metadata handling)
'''
setup(
	name='cubvh', # package name, import this to use python API
	version='0.1.0',
	description='CUDA BVH implementation',
	url='https://github.com/ashawkey/cubvh',
	author='kiui',
	author_email='ashawkey1999@gmail.com',
	packages=['cubvh'],
	ext_modules=[
		CUDAExtension(
			name='_cubvh', # extension name, import this to use CUDA API
			sources=[os.path.join(_src_path, 'src', f) for f in [
				'bvh.cu',
				'api.cu',
				'bindings.cpp',
			]],
			include_dirs=[
				os.path.join(_src_path, 'include'),
				find_eigen(),
			],
			extra_compile_args={
				'cxx': base_cflags,
				'nvcc': base_nvcc_flags,
			}
		),
	],
	cmdclass={
		'build_ext': BuildExtension,
	},
	install_requires=[
		'ninja',
		'pybind11',
		'trimesh',
		'torch',
		'numpy',
	],
)
