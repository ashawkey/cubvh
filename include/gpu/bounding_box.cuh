#pragma once

#include <gpu/common.h>
#include <gpu/triangle.cuh>

namespace cubvh {

// POD layout (min, max) is a serialization contract; do not reorder.
struct BoundingBox {
    Eigen::Vector3f min, max;

    __host__ __device__ BoundingBox()
        : min(Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity())),
          max(Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity())) {}

    BoundingBox(std::vector<Triangle>::iterator begin, std::vector<Triangle>::iterator end) : BoundingBox() {
        for (auto it = begin; it != end; ++it) {
            enlarge(*it);
        }
    }

    __host__ __device__ void enlarge(const Eigen::Vector3f& p) {
        min = min.cwiseMin(p);
        max = max.cwiseMax(p);
    }

    __host__ __device__ void enlarge(const Triangle& t) {
        enlarge(t.a);
        enlarge(t.b);
        enlarge(t.c);
    }

    __host__ __device__ float distance_sq(const Eigen::Vector3f& p) const {
        float d2 = 0.0f;
        for (int axis = 0; axis < 3; ++axis) {
            const float excess = fmaxf(fmaxf(min[axis] - p[axis], p[axis] - max[axis]), 0.0f);
            d2 += excess * excess;
        }
        return d2;
    }

    // Slab method (Kay-Kajiya). Returns {tmin, tmax}, unclamped, so a box
    // containing or behind the origin reports a negative tmin; on an empty
    // interval both entries are FLT_MAX. safe_divide keeps axis-parallel
    // rays finite.
    __host__ __device__ Eigen::Vector2f ray_intersect(Eigen::Ref<const Eigen::Vector3f> ro, Eigen::Ref<const Eigen::Vector3f> rd) const {
        float t_enter = -std::numeric_limits<float>::max();
        float t_exit = std::numeric_limits<float>::max();

        #pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            const float lo = safe_divide(min[axis] - ro[axis], rd[axis]);
            const float hi = safe_divide(max[axis] - ro[axis], rd[axis]);
            t_enter = fmaxf(t_enter, fminf(lo, hi));
            t_exit = fminf(t_exit, fmaxf(lo, hi));
        }

        if (t_exit < t_enter) {
            return Eigen::Vector2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        }
        return Eigen::Vector2f(t_enter, t_exit);
    }
};

static_assert(sizeof(BoundingBox) == 24, "BoundingBox layout is a serialization contract");

} // namespace cubvh
