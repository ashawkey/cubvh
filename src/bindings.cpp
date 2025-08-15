// #include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <cubvh/api.h>
#include <cpu/api_cpu.h>


namespace py = pybind11;
using namespace cubvh;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

// CUDA API
py::class_<cuBVH>(m, "cuBVH")
    .def("ray_trace", &cuBVH::ray_trace)
    .def("unsigned_distance", &cuBVH::unsigned_distance)
    .def("signed_distance", &cuBVH::signed_distance);

m.def("create_cuBVH", &create_cuBVH);

m.def("floodfill", &floodfill);

m.def("sparse_marching_cubes", &sparse_marching_cubes, py::arg("coords"), py::arg("corners"), py::arg("iso"), py::arg("ensure_consistency") = false);

// CPU API
m.def("fill_holes", &fill_holes);
m.def("merge_vertices", &merge_vertices, py::arg("vertices"), py::arg("faces"), py::arg("threshold"));
m.def("sparse_marching_cubes_cpu", &sparse_marching_cubes_cpu,
    py::arg("coords"), py::arg("corners"), py::arg("iso"), py::arg("ensure_consistency") = false);

}