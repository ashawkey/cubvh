#pragma once

#include <gpu/common.h>
#include <gpu/triangle.cuh>
#include <gpu/bounding_box.cuh>
#include <gpu/gpu_memory.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include <torch/torch.h>

namespace cubvh {

// Node encoding (a serialization contract: previously saved node arrays
// must keep loading and traversing identically):
//   left_idx <  0: leaf covering triangle indices [-left_idx-1, -right_idx-1)
//   left_idx >= 0: inner node whose children are nodes [left_idx, right_idx)
//   escape_idx: next node in pre-order after this node's subtree; -1 ends
//   the traversal. Node 0 is the root.
struct TriangleBvhNode {
    BoundingBox bb;
    int left_idx;
    int right_idx;
    int escape_idx;
};

static_assert(sizeof(TriangleBvhNode) == 36, "TriangleBvhNode layout is a serialization contract");

// Fixed-capacity LIFO. A push beyond capacity is dropped and sets a sticky
// overflow flag (warned once per instance); callers fall back to a
// stackless traversal when that happens.
template <typename T, int MAX_SIZE = 32>
class FixedStack {
public:
    __host__ __device__ void push(T value) {
        if (m_size >= MAX_SIZE) {
            if (!m_overflowed) {
                m_overflowed = true;
                printf("warning: FixedStack exceeded its capacity of %d\n", MAX_SIZE);
            }
            return;
        }
        m_data[m_size++] = value;
    }

    __host__ __device__ T pop() {
        return m_data[--m_size];
    }

    __host__ __device__ bool empty() const {
        return m_size == 0;
    }

    __host__ __device__ bool overflowed() const {
        return m_overflowed;
    }

private:
    T m_data[MAX_SIZE];
    int m_size = 0;
    bool m_overflowed = false;
};

using FixedIntStack = FixedStack<int>;
using FixedIntStackLarge = FixedStack<int, 64>;

class TriangleBvh {
public:
    virtual ~TriangleBvh() {}

    virtual void build(std::vector<Triangle>& triangles, uint32_t n_primitives_per_leaf) = 0;
    virtual void signed_distance_gpu(uint32_t n_elements, uint32_t mode, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, cudaStream_t stream) = 0;
    virtual void unsigned_distance_gpu(uint32_t n_elements, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, cudaStream_t stream) = 0;
    virtual void ray_trace_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d, float* positions, int64_t* face_id, float* depth, const Triangle* gpu_triangles, cudaStream_t stream) = 0;

    static std::unique_ptr<TriangleBvh> make();

    TriangleBvhNode* nodes_gpu() const { return m_nodes_gpu.data(); }

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

protected:
    TriangleBvh() {}

    std::vector<TriangleBvhNode> m_nodes;
    GPUMemory<TriangleBvhNode> m_nodes_gpu;
};

} // namespace cubvh
