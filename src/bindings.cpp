// #include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <cubvh/api.h>


namespace py = pybind11;
using namespace cubvh;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

py::class_<cuBVH>(m, "cuBVH")
    .def("ray_trace", &cuBVH::ray_trace)
    .def("unsigned_distance", &cuBVH::unsigned_distance)
    .def("signed_distance", &cuBVH::signed_distance);

m.def("create_cuBVH", &create_cuBVH);

m.def("floodfill", &floodfill);

}