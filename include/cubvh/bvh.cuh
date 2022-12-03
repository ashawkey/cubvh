#pragma once

#include <cubvh/common.h>
#include <cubvh/triangle.cuh>
#include <cubvh/bounding_box.cuh>
#include <cubvh/gpu_memory.h>

#include <memory>

namespace cubvh {

struct TriangleBvhNode {
    BoundingBox bb;
    int left_idx; // negative values indicate leaves
    int right_idx;
};


template <typename T, int MAX_SIZE=32>
class FixedStack {
public:
    __host__ __device__ void push(T val) {
        if (m_count >= MAX_SIZE-1) {
            printf("WARNING TOO BIG\n");
        }
        m_elems[m_count++] = val;
    }

    __host__ __device__ T pop() {
        return m_elems[--m_count];
    }

    __host__ __device__ bool empty() const {
        return m_count <= 0;
    }

private:
    T m_elems[MAX_SIZE];
    int m_count = 0;
};

using FixedIntStack = FixedStack<int>;


class TriangleBvh {

protected:
    std::vector<TriangleBvhNode> m_nodes;
    GPUMemory<TriangleBvhNode> m_nodes_gpu;
    TriangleBvh() {};

public:
    virtual void build(std::vector<Triangle>& triangles, uint32_t n_primitives_per_leaf) = 0;

    virtual void unsigned_distance_gpu(uint32_t n_elements, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, cudaStream_t stream) = 0;
    virtual void ray_trace_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d, float* positions, int64_t* face_id, float* depth, const Triangle* gpu_triangles, cudaStream_t stream) = 0;

    // KIUI: not supported now.
    // virtual void signed_distance_gpu(uint32_t n_elements, EMeshSdfMode mode, const Eigen::Vector3f* gpu_positions, float* gpu_distances, const Triangle* gpu_triangles, bool use_existing_distances_as_upper_bounds, cudaStream_t stream) = 0;
    // virtual bool touches_triangle(const BoundingBox& bb, const Triangle* __restrict__ triangles) const = 0;
    // virtual void build_optix(const GPUMemory<Triangle>& triangles, cudaStream_t stream) = 0;

    static std::unique_ptr<TriangleBvh> make();

    TriangleBvhNode* nodes_gpu() const {
        return m_nodes_gpu.data();
    }

};

}