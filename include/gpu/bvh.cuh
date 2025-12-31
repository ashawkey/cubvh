#pragma once

#include <gpu/common.h>
#include <gpu/triangle.cuh>
#include <gpu/bounding_box.cuh>
#include <gpu/gpu_memory.h>

#include <memory>

#include <torch/torch.h>

namespace cubvh {

struct TriangleBvhNode {
    BoundingBox bb;
    int left_idx; // negative values indicate leaves
    int right_idx;
    // Threaded BVH escape index for stackless traversal: index of the next node
    // to visit after finishing this node's subtree. -1 indicates termination.
    int escape_idx;
};


template <typename T, int MAX_SIZE=32>
class FixedStack {
public:
    __host__ __device__ void push(T val) {
        // If overflowing, flag and drop the push; a stackless fallback will be used.
        if (m_count >= MAX_SIZE) {
            if (!m_overflowed) {
                printf("WARNING TOO BIG (stack overflow)\n");
            }
            m_overflowed = true;
            return;
        }
        m_elems[m_count++] = val;
    }

    __host__ __device__ T pop() {
        return m_elems[--m_count];
    }

    __host__ __device__ bool empty() const {
        return m_count <= 0;
    }

    __host__ __device__ bool overflowed() const {
        return m_overflowed;
    }

private:
    T m_elems[MAX_SIZE];
    int m_count = 0;
    bool m_overflowed = false;
};

using FixedIntStack = FixedStack<int>;


class TriangleBvh {

protected:
    std::vector<TriangleBvhNode> m_nodes;
    GPUMemory<TriangleBvhNode> m_nodes_gpu;
    TriangleBvh() {};

public:
    virtual void build(std::vector<Triangle>& triangles, uint32_t n_primitives_per_leaf) = 0;

    virtual void signed_distance_gpu(uint32_t n_elements, uint32_t mode, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, cudaStream_t stream) = 0;
    virtual void unsigned_distance_gpu(uint32_t n_elements, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, cudaStream_t stream) = 0;
    virtual void ray_trace_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d, float* positions, int64_t* face_id, float* depth, const Triangle* gpu_triangles, cudaStream_t stream) = 0;

    // KIUI: not supported now.
    // virtual bool touches_triangle(const BoundingBox& bb, const Triangle* __restrict__ triangles) const = 0;
    // virtual void build_optix(const GPUMemory<Triangle>& triangles, cudaStream_t stream) = 0;

    static std::unique_ptr<TriangleBvh> make();

    TriangleBvhNode* nodes_gpu() const {
        return m_nodes_gpu.data();
    }

    std::unordered_map<std::string, at::Tensor> state_dict() const {
        std::unordered_map<std::string, at::Tensor> state;
        // Create a tensor from the CPU data
        auto nodes_tensor = at::from_blob(
            (void*)m_nodes.data(),
            {static_cast<int64_t>(m_nodes.size()), static_cast<int64_t>(sizeof(TriangleBvhNode) / sizeof(int32_t))},
            at::TensorOptions().dtype(at::kInt).device(at::kCPU)
        ).clone();
        state["nodes"] = nodes_tensor;
        return state;
    }

    void load_state_dict(const std::unordered_map<std::string, at::Tensor>& state) {
        // If nodes exist in the state, load them
        auto it = state.find("nodes");
        if (it == state.end()) {
            throw std::runtime_error("State dict does not contain 'nodes'");
        }
        const at::Tensor& nodes_tensor = it->second;
        // Resize m_nodes to match the size of the tensor
        m_nodes.resize(nodes_tensor.size(0));
        // Copy data from the tensor to m_nodes
        std::memcpy(m_nodes.data(), nodes_tensor.data_ptr(), m_nodes.size() * sizeof(TriangleBvhNode));
        // Removed for it is now done lazily
        // m_nodes_gpu.resize_and_copy_from_host(m_nodes);
    }
};

}