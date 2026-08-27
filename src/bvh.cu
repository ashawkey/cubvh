#include <gpu/bvh.cuh>
#include <gpu/pcg32.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace cubvh {

using namespace Eigen;

constexpr float MAX_DIST = 1000.0f;
constexpr float MAX_DIST_SQ = MAX_DIST * MAX_DIST;

// squared distance under which a point counts as lying on a surface element
constexpr float COINCIDENT_EPS_SQ = 1e-6f;

constexpr uint32_t N_STAB_DIRS = 32;

// 4-wide BVH. Queries come in two flavors: a fixed-stack traversal (fast
// path) and a stackless walk along the pre-order escape links, which serves
// as the overflow fallback and handles any well-formed node array.
class TriangleBvhImpl : public TriangleBvh {
public:
    static constexpr int FANOUT = 4;

    TriangleBvhImpl() {}

    // Top-down median-split build (PBRT ch. 4 family): each node is split
    // twice around the centroid median on the axis of maximum centroid
    // variance, giving 4 contiguous sub-ranges of the (reordered) triangle
    // vector. Children occupy contiguous node slots.
    void build(std::vector<Triangle>& triangles, uint32_t n_primitives_per_leaf) override {
        m_nodes.clear();
        m_nodes.emplace_back();
        m_nodes.front().bb = BoundingBox(triangles.begin(), triangles.end());

        struct SplitJob {
            int node;
            TriIt begin;
            TriIt end;
        };
        std::vector<SplitJob> pending;
        pending.push_back({0, triangles.begin(), triangles.end()});

        while (!pending.empty()) {
            const SplitJob job = pending.back();
            pending.pop_back();

            TriIt cut[FANOUT + 1];
            cut[0] = job.begin;
            cut[4] = job.end;
            cut[2] = median_split(cut[0], cut[4]);
            cut[1] = median_split(cut[0], cut[2]);
            cut[3] = median_split(cut[2], cut[4]);

            const int first_child = (int)m_nodes.size();
            m_nodes[job.node].left_idx = first_child;
            m_nodes[job.node].right_idx = first_child + FANOUT;

            for (int k = 0; k < FANOUT; ++k) {
                TriangleBvhNode child;
                child.bb = BoundingBox(cut[k], cut[k + 1]);
                if ((uint32_t)(cut[k + 1] - cut[k]) <= n_primitives_per_leaf) {
                    child.left_idx = -(int)(cut[k] - triangles.begin()) - 1;
                    child.right_idx = -(int)(cut[k + 1] - triangles.begin()) - 1;
                } else {
                    pending.push_back({first_child + k, cut[k], cut[k + 1]});
                }
                m_nodes.push_back(child);
            }
        }

        link_escapes(0, -1);
    }

    static __host__ __device__ int leaf_begin(const TriangleBvhNode& node) {
        return -node.left_idx - 1;
    }

    static __host__ __device__ int leaf_end(const TriangleBvhNode& node) {
        return -node.right_idx - 1;
    }

    static __host__ __device__ void order_ascending(float& ka, int& sa, float& kb, int& sb) {
        if (ka > kb) {
            const float kt = ka; ka = kb; kb = kt;
            const int st = sa; sa = sb; sb = st;
        }
    }

    // 5-exchange sorting network: keys ascending afterwards
    static __host__ __device__ void sort_children(float key[FANOUT], int slot[FANOUT]) {
        order_ascending(key[0], slot[0], key[1], slot[1]);
        order_ascending(key[2], slot[2], key[3], slot[3]);
        order_ascending(key[0], slot[0], key[2], slot[2]);
        order_ascending(key[1], slot[1], key[3], slot[3]);
        order_ascending(key[1], slot[1], key[2], slot[2]);
    }

    // Stackless walk along the escape links (textbook "ropes").
    static __host__ __device__ std::pair<int, float> ray_intersect_stackless(Ref<const Vector3f> ro, Ref<const Vector3f> rd, const TriangleBvhNode* nodes, const Triangle* triangles) {
        int hit = -1;
        float t_hit = MAX_DIST;
        int idx = 0;
        while (idx >= 0) {
            const TriangleBvhNode& node = nodes[idx];
            if (node.bb.ray_intersect(ro, rd).x() >= t_hit) {
                idx = node.escape_idx;
            } else if (node.left_idx < 0) {
                for (int i = leaf_begin(node); i < leaf_end(node); ++i) {
                    const float t = triangles[i].ray_intersect(ro, rd);
                    if (t < t_hit) {
                        t_hit = t;
                        hit = i;
                    }
                }
                idx = node.escape_idx;
            } else {
                idx = node.left_idx; // siblings follow via their escape links
            }
        }
        return std::make_pair(hit, t_hit);
    }

    // Returns {triangle array index, ray parameter}; {-1, MAX_DIST} on miss.
    static __host__ __device__ std::pair<int, float> ray_intersect(Ref<const Vector3f> ro, Ref<const Vector3f> rd, const TriangleBvhNode* nodes, const Triangle* triangles) {
        FixedIntStack stack;
        stack.push(0);

        int hit = -1;
        float t_hit = MAX_DIST;

        while (!stack.empty()) {
            const TriangleBvhNode& node = nodes[stack.pop()];

            if (node.left_idx < 0) {
                for (int i = leaf_begin(node); i < leaf_end(node); ++i) {
                    const float t = triangles[i].ray_intersect(ro, rd);
                    if (t < t_hit) {
                        t_hit = t;
                        hit = i;
                    }
                }
            } else {
                if (node.right_idx - node.left_idx != FANOUT) {
                    // unexpected fan-out; the stackless walk handles any layout
                    return ray_intersect_stackless(ro, rd, nodes, triangles);
                }
                // order all four children by entry distance, then push the
                // surviving ones far-to-near so the nearest pops first
                int slot[FANOUT];
                float key[FANOUT];
                // Rolled deliberately: unrolling keeps all four children's
                // boxes and hit distances live at once, which costs enough
                // registers to halve occupancy on wave32 parts.
                #pragma unroll 1
                for (int k = 0; k < FANOUT; ++k) {
                    slot[k] = node.left_idx + k;
                    key[k] = nodes[slot[k]].bb.ray_intersect(ro, rd).x();
                }
                sort_children(key, slot);
                #pragma unroll 1
                for (int k = FANOUT - 1; k >= 0; --k) {
                    if (key[k] < t_hit) {
                        stack.push(slot[k]);
                    }
                }
            }

            if (stack.overflowed()) {
                return ray_intersect_stackless(ro, rd, nodes, triangles);
            }
        }

        return std::make_pair(hit, t_hit);
    }

    static __host__ __device__ std::pair<int, float> closest_result(int idx, float d2) {
        if (idx < 0) {
            return std::make_pair(-1, std::numeric_limits<float>::infinity());
        }
        return std::make_pair(idx, sqrtf(d2));
    }

    static __host__ __device__ std::pair<int, float> closest_triangle_stackless(const Vector3f& point, const TriangleBvhNode* nodes, const Triangle* triangles, float max_distance_sq) {
        int best = -1;
        float d2_best = max_distance_sq;
        int idx = 0;
        while (idx >= 0) {
            const TriangleBvhNode& node = nodes[idx];
            if (node.bb.distance_sq(point) > d2_best) {
                idx = node.escape_idx;
            } else if (node.left_idx < 0) {
                for (int i = leaf_begin(node); i < leaf_end(node); ++i) {
                    const float d2 = triangles[i].distance_sq(point);
                    if (d2 <= d2_best) {
                        d2_best = d2;
                        best = i;
                    }
                }
                idx = node.escape_idx;
            } else {
                idx = node.left_idx;
            }
        }
        return closest_result(best, d2_best);
    }

    // Returns {triangle array index, unsquared distance}. When nothing lies
    // within the bound the result is {-1, +inf}.
    static __host__ __device__ std::pair<int, float> closest_triangle(const Vector3f& point, const TriangleBvhNode* nodes, const Triangle* triangles, float max_distance_sq = MAX_DIST_SQ) {
        FixedIntStackLarge stack;
        stack.push(0);

        int best = -1;
        float d2_best = max_distance_sq;

        while (!stack.empty()) {
            const TriangleBvhNode& node = nodes[stack.pop()];

            if (node.left_idx < 0) {
                for (int i = leaf_begin(node); i < leaf_end(node); ++i) {
                    const float d2 = triangles[i].distance_sq(point);
                    // non-strict: an exact tie goes to the later triangle;
                    // degenerate triangles report inf and never win
                    if (d2 <= d2_best) {
                        d2_best = d2;
                        best = i;
                    }
                }
            } else {
                if (node.right_idx - node.left_idx != FANOUT) {
                    return closest_triangle_stackless(point, nodes, triangles, d2_best);
                }
                int slot[FANOUT];
                float key[FANOUT];
                #pragma unroll
                for (int k = 0; k < FANOUT; ++k) {
                    slot[k] = node.left_idx + k;
                    key[k] = nodes[slot[k]].bb.distance_sq(point);
                }
                sort_children(key, slot);
                #pragma unroll
                for (int k = FANOUT - 1; k >= 0; --k) {
                    if (key[k] <= d2_best) {
                        stack.push(slot[k]);
                    }
                }
            }

            if (stack.overflowed()) {
                // restart with the bound found so far; non-strict acceptance
                // re-finds the same winner
                return closest_triangle_stackless(point, nodes, triangles, d2_best);
            }
        }

        return closest_result(best, d2_best);
    }

    static __host__ __device__ Vector3f avg_normal_stackless(const Vector3f& point, const TriangleBvhNode* nodes, const Triangle* triangles) {
        Vector3f sum = Vector3f::Zero();
        float weight = 0.0f;
        int idx = 0;
        while (idx >= 0) {
            const TriangleBvhNode& node = nodes[idx];
            if (node.bb.distance_sq(point) >= COINCIDENT_EPS_SQ) {
                idx = node.escape_idx;
            } else if (node.left_idx < 0) {
                for (int i = leaf_begin(node); i < leaf_end(node); ++i) {
                    if (triangles[i].distance_sq(point) < COINCIDENT_EPS_SQ) {
                        sum += triangles[i].normal();
                        weight += 1.0f;
                    }
                }
                idx = node.escape_idx;
            } else {
                idx = node.left_idx;
            }
        }
        return sum / weight;
    }

    // Unnormalized average of the unit normals of every triangle passing
    // through `point` (assumed to lie on the surface); NaN vector when none
    // is within epsilon.
    static __host__ __device__ Vector3f avg_normal_around_point(const Vector3f& point, const TriangleBvhNode* nodes, const Triangle* triangles) {
        FixedIntStack stack;
        stack.push(0);

        Vector3f sum = Vector3f::Zero();
        float weight = 0.0f;

        while (!stack.empty()) {
            const TriangleBvhNode& node = nodes[stack.pop()];

            if (node.left_idx < 0) {
                for (int i = leaf_begin(node); i < leaf_end(node); ++i) {
                    if (triangles[i].distance_sq(point) < COINCIDENT_EPS_SQ) {
                        sum += triangles[i].normal();
                        weight += 1.0f;
                    }
                }
            } else {
                if (node.right_idx - node.left_idx != FANOUT) {
                    return avg_normal_stackless(point, nodes, triangles);
                }
                #pragma unroll
                for (int k = 0; k < FANOUT; ++k) {
                    const int c = node.left_idx + k;
                    if (nodes[c].bb.distance_sq(point) < COINCIDENT_EPS_SQ) {
                        stack.push(c);
                    }
                }
            }

            if (stack.overflowed()) {
                return avg_normal_stackless(point, nodes, triangles);
            }
        }

        return sum / weight;
    }

    // Sign from the average normal at the closest surface point.
    static __host__ __device__ std::pair<int, float> signed_distance_watertight(const Vector3f& point, const TriangleBvhNode* nodes, const Triangle* triangles, float max_distance_sq = MAX_DIST_SQ) {
        const std::pair<int, float> best = closest_triangle(point, nodes, triangles, max_distance_sq);
        if (best.first < 0) {
            return best;
        }
        const Vector3f surface = triangles[best.first].closest_point(point);
        const Vector3f n = avg_normal_around_point(surface, nodes, triangles);
        return std::make_pair(best.first, copysignf(best.second, n.dot(point - surface)));
    }

    // Sign by ray stabbing (Nooruddin-Turk) over a randomized Fibonacci
    // direction set: any unobstructed direction (either way) means outside.
    static __host__ __device__ std::pair<int, float> signed_distance_raystab(const Vector3f& point, const TriangleBvhNode* nodes, const Triangle* triangles, float max_distance_sq = MAX_DIST_SQ, pcg32 rng = {}) {
        const std::pair<int, float> best = closest_triangle(point, nodes, triangles, max_distance_sq);
        if (best.first < 0) {
            return best;
        }

        const Vector2f offset = {rng.next_float(), rng.next_float()};
        for (uint32_t i = 0; i < N_STAB_DIRS; ++i) {
            const Vector3f dir = fibonacci_dir<N_STAB_DIRS>(i, offset);
            const Vector3f neg_dir = -dir;
            if (ray_intersect(point, neg_dir, nodes, triangles).first < 0 ||
                ray_intersect(point, dir, nodes, triangles).first < 0) {
                return best; // sees open space in some direction: outside
            }
        }

        return std::make_pair(best.first, -best.second);
    }

    void signed_distance_gpu(uint32_t n_elements, uint32_t mode, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, cudaStream_t stream) override;
    void unsigned_distance_gpu(uint32_t n_elements, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, cudaStream_t stream) override;
    void ray_trace_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d, float* positions, int64_t* face_id, float* depth, const Triangle* gpu_triangles, cudaStream_t stream) override;

private:
    using TriIt = std::vector<Triangle>::iterator;

    // Partition [begin, end) around the centroid median on the axis of
    // maximum centroid variance; returns the median iterator.
    static TriIt median_split(TriIt begin, TriIt end) {
        Vector3f mean = Vector3f::Zero();
        for (TriIt it = begin; it != end; ++it) {
            mean += it->centroid();
        }
        mean /= (float)(end - begin);

        Vector3f variance = Vector3f::Zero();
        for (TriIt it = begin; it != end; ++it) {
            const Vector3f d = it->centroid() - mean;
            variance += d.cwiseProduct(d);
        }

        int axis = 0;
        variance.maxCoeff(&axis);

        const TriIt median = begin + (end - begin) / 2;
        std::nth_element(begin, median, end, [axis](const Triangle& lhs, const Triangle& rhs) {
            return lhs.centroid(axis) < rhs.centroid(axis);
        });
        return median;
    }

    // escape_idx = next pre-order node after the subtree: the next sibling
    // if any, else the parent's escape; -1 terminates.
    void link_escapes(int node, int escape) {
        m_nodes[node].escape_idx = escape;
        if (m_nodes[node].left_idx < 0) {
            return;
        }
        const int last = m_nodes[node].right_idx;
        for (int c = m_nodes[node].left_idx; c < last; ++c) {
            link_escapes(c, c + 1 < last ? c + 1 : escape);
        }
    }

    void ensure_nodes_on_gpu() {
        // deliberately lazy: deserialized trees may never need the GPU copy
        if (m_nodes_gpu.data() == nullptr) {
            m_nodes_gpu.resize_and_copy_from_host(m_nodes);
        }
    }
};

__global__ void bvh_ray_trace_kernel(const uint32_t n, const Vector3f* __restrict__ rays_o, const Vector3f* __restrict__ rays_d, Vector3f* __restrict__ positions, int64_t* __restrict__ face_id, float* __restrict__ depth, const TriangleBvhNode* __restrict__ nodes, const Triangle* __restrict__ triangles) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    const Vector3f ro = rays_o[i];
    const Vector3f rd = rays_d[i];

    const std::pair<int, float> hit = TriangleBvhImpl::ray_intersect(ro, rd, nodes, triangles);

    // a miss lands at distance MAX_DIST along the ray
    positions[i] = ro + hit.second * rd;
    depth[i] = hit.second;
    face_id[i] = hit.first >= 0 ? triangles[hit.first].id : -1;
}

__global__ void bvh_unsigned_distance_kernel(const uint32_t n, const Vector3f* __restrict__ positions, float* __restrict__ distances, int64_t* __restrict__ face_id, Vector3f* __restrict__ uvw, const TriangleBvhNode* __restrict__ nodes, const Triangle* __restrict__ triangles, bool use_existing_distances_as_upper_bounds) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    float max_distance_sq = MAX_DIST_SQ;
    if (use_existing_distances_as_upper_bounds) {
        max_distance_sq = distances[i] * distances[i];
    }

    const Vector3f point = positions[i];
    const std::pair<int, float> best = TriangleBvhImpl::closest_triangle(point, nodes, triangles, max_distance_sq);
    if (best.first < 0) {
        distances[i] = best.second;
        face_id[i] = -1;
        if (uvw) {
            uvw[i] = Vector3f::Zero();
        }
        return;
    }
    const Triangle& tri = triangles[best.first];

    distances[i] = best.second;
    face_id[i] = tri.id;
    if (uvw) {
        uvw[i] = tri.barycentric(tri.closest_point(point));
    }
}

__global__ void bvh_signed_distance_watertight_kernel(const uint32_t n, const Vector3f* __restrict__ positions, float* __restrict__ distances, int64_t* __restrict__ face_id, Vector3f* __restrict__ uvw, const TriangleBvhNode* __restrict__ nodes, const Triangle* __restrict__ triangles) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    const Vector3f point = positions[i];
    const std::pair<int, float> best = TriangleBvhImpl::signed_distance_watertight(point, nodes, triangles);
    if (best.first < 0) {
        distances[i] = best.second;
        face_id[i] = -1;
        if (uvw) {
            uvw[i] = Vector3f::Zero();
        }
        return;
    }
    const Triangle& tri = triangles[best.first];

    distances[i] = best.second;
    face_id[i] = tri.id;
    if (uvw) {
        uvw[i] = tri.barycentric(tri.closest_point(point));
    }
}

__global__ void bvh_signed_distance_raystab_kernel(const uint32_t n, const Vector3f* __restrict__ positions, float* __restrict__ distances, int64_t* __restrict__ face_id, Vector3f* __restrict__ uvw, const TriangleBvhNode* __restrict__ nodes, const Triangle* __restrict__ triangles) {
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    // per-element stream offset: 2 draws per element
    pcg32 rng;
    rng.advance(i * 2);

    const Vector3f point = positions[i];
    const std::pair<int, float> best = TriangleBvhImpl::signed_distance_raystab(point, nodes, triangles, MAX_DIST_SQ, rng);
    if (best.first < 0) {
        distances[i] = best.second;
        face_id[i] = -1;
        if (uvw) {
            uvw[i] = Vector3f::Zero();
        }
        return;
    }
    const Triangle& tri = triangles[best.first];

    distances[i] = best.second;
    face_id[i] = tri.id;
    if (uvw) {
        uvw[i] = tri.barycentric(tri.closest_point(point));
    }
}

void TriangleBvhImpl::signed_distance_gpu(uint32_t n_elements, uint32_t mode, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, cudaStream_t stream) {
    ensure_nodes_on_gpu();
    if (mode == 0) {
        linear_kernel(bvh_signed_distance_watertight_kernel, 0u, stream, n_elements,
            (const Vector3f*)positions, distances, face_id, (Vector3f*)uvw, m_nodes_gpu.data(), gpu_triangles);
    } else {
        linear_kernel(bvh_signed_distance_raystab_kernel, 0u, stream, n_elements,
            (const Vector3f*)positions, distances, face_id, (Vector3f*)uvw, m_nodes_gpu.data(), gpu_triangles);
    }
}

void TriangleBvhImpl::unsigned_distance_gpu(uint32_t n_elements, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, cudaStream_t stream) {
    ensure_nodes_on_gpu();
    linear_kernel(bvh_unsigned_distance_kernel, 0u, stream, n_elements,
        (const Vector3f*)positions, distances, face_id, (Vector3f*)uvw, m_nodes_gpu.data(), gpu_triangles, false);
}

void TriangleBvhImpl::ray_trace_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d, float* positions, int64_t* face_id, float* depth, const Triangle* gpu_triangles, cudaStream_t stream) {
    ensure_nodes_on_gpu();
    linear_kernel(bvh_ray_trace_kernel, 0u, stream, n_elements,
        (const Vector3f*)rays_o, (const Vector3f*)rays_d, (Vector3f*)positions, face_id, depth, m_nodes_gpu.data(), gpu_triangles);
}

std::unique_ptr<TriangleBvh> TriangleBvh::make() {
    return std::unique_ptr<TriangleBvh>(new TriangleBvhImpl{});
}

} // namespace cubvh
