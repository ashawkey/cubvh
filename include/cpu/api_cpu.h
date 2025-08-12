#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <cpu/fill_holes.h>
#include <cpu/merge_vertices.h>

#include <vector>

namespace py = pybind11;

namespace cubvh {

static py::array_t<int> fill_holes(
    py::array_t<float, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<int,   py::array::c_style | py::array::forcecast> faces,
    bool return_added,
    bool check_containment,
    double eps_d,
    bool verbose) {
    // Validate shapes
    auto vbuf = vertices.request();
    auto fbuf = faces.request();
    if (!(vbuf.ndim == 2 && vbuf.shape[1] == 3)) {
        throw std::runtime_error("vertices must be of shape [N,3]");
    }
    if (!(fbuf.ndim == 2 && fbuf.shape[1] == 3)) {
        throw std::runtime_error("faces must be of shape [M,3]");
    }

    const size_t N = static_cast<size_t>(vbuf.shape[0]);
    const size_t M = static_cast<size_t>(fbuf.shape[0]);

    const float* vptr = static_cast<const float*>(vbuf.ptr);
    const int*   fptr = static_cast<const int*>(fbuf.ptr);

    std::vector<Eigen::Vector3f> V(N);
    for (size_t i = 0; i < N; ++i) {
        V[i] = Eigen::Vector3f(vptr[3*i+0], vptr[3*i+1], vptr[3*i+2]);
    }

    std::vector<Eigen::Vector3i> F(M);
    for (size_t i = 0; i < M; ++i) {
        F[i] = Eigen::Vector3i(fptr[3*i+0], fptr[3*i+1], fptr[3*i+2]);
    }

    cubvh::cpu::HoleFillOptions opt;
    opt.checkContainment = check_containment;
    opt.eps = static_cast<float>(eps_d);
    opt.verbose = verbose;

    if (return_added) {
        auto added = cubvh::cpu::fill_holes(V, F, opt);
        py::array_t<int> out({(py::ssize_t)added.size(), (py::ssize_t)3});
        auto obuf = out.request();
        int* optr = static_cast<int*>(obuf.ptr);
        for (size_t i = 0; i < added.size(); ++i) {
            optr[3*i+0] = added[i][0];
            optr[3*i+1] = added[i][1];
            optr[3*i+2] = added[i][2];
        }
        return out;
    } else {
        cubvh::cpu::fill_holes_inplace(V, F, opt);
        py::array_t<int> out({(py::ssize_t)F.size(), (py::ssize_t)3});
        auto obuf = out.request();
        int* optr = static_cast<int*>(obuf.ptr);
        for (size_t i = 0; i < F.size(); ++i) {
            optr[3*i+0] = F[i][0];
            optr[3*i+1] = F[i][1];
            optr[3*i+2] = F[i][2];
        }
        return out;
    }
}

// merge_vertices binding: returns (vertices, faces) after merge
static std::pair<py::array_t<float>, py::array_t<int>> merge_vertices(
    py::array_t<float, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<int,   py::array::c_style | py::array::forcecast> faces,
    double threshold_d) {
    auto vbuf = vertices.request();
    auto fbuf = faces.request();
    if (!(vbuf.ndim == 2 && vbuf.shape[1] == 3)) {
        throw std::runtime_error("vertices must be of shape [N,3]");
    }
    if (!(fbuf.ndim == 2 && fbuf.shape[1] == 3)) {
        throw std::runtime_error("faces must be of shape [M,3]");
    }
    const size_t N = static_cast<size_t>(vbuf.shape[0]);
    const size_t M = static_cast<size_t>(fbuf.shape[0]);
    const float* vptr = static_cast<const float*>(vbuf.ptr);
    const int* fptr = static_cast<const int*>(fbuf.ptr);

    std::vector<Eigen::Vector3f> V(N);
    for (size_t i=0;i<N;++i) {
        V[i] = Eigen::Vector3f(vptr[3*i+0], vptr[3*i+1], vptr[3*i+2]);
    }
    std::vector<Eigen::Vector3i> F(M);
    for (size_t i=0;i<M;++i) {
        F[i] = Eigen::Vector3i(fptr[3*i+0], fptr[3*i+1], fptr[3*i+2]);
    }

    std::vector<Eigen::Vector3f> V_out; std::vector<Eigen::Vector3i> F_out;
    cubvh::cpu::merge_vertices(V, F, static_cast<float>(threshold_d), V_out, F_out);

    py::array_t<float> v_out({(py::ssize_t)V_out.size(), (py::ssize_t)3});
    py::array_t<int> f_out({(py::ssize_t)F_out.size(), (py::ssize_t)3});
    auto vObuf = v_out.request(); auto fObuf = f_out.request();
    float* vO = static_cast<float*>(vObuf.ptr);
    int* fO = static_cast<int*>(fObuf.ptr);
    for (size_t i=0;i<V_out.size();++i) {
        vO[3*i+0] = V_out[i][0]; vO[3*i+1] = V_out[i][1]; vO[3*i+2] = V_out[i][2];
    }
    for (size_t i=0;i<F_out.size();++i) {
        fO[3*i+0] = F_out[i][0]; fO[3*i+1] = F_out[i][1]; fO[3*i+2] = F_out[i][2];
    }
    return {v_out, f_out};
}

} // namespace cubvh
