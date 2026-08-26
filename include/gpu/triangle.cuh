#pragma once

#include <gpu/common.h>

namespace cubvh {

// POD layout (a, b, c, id) is a serialization contract; do not reorder.
struct Triangle {
    Eigen::Vector3f a, b, c;
    int64_t id;

    __host__ __device__ Eigen::Vector3f centroid() const {
        return (a + b + c) / 3.0f;
    }

    __host__ __device__ float centroid(int axis) const {
        return (a[axis] + b[axis] + c[axis]) / 3.0f;
    }

    // Unit normal along cross(b-a, c-a); NaN vector for zero-area triangles.
    __host__ __device__ Eigen::Vector3f normal() const {
        const Eigen::Vector3f n = (b - a).cross(c - a);
        return n / n.norm();
    }

    // Double-sided Moller-Trumbore. Returns the ray parameter t >= 0 of the
    // hit, or 1e6 on a miss. safe_divide guards the reciprocal determinant,
    // so near-parallel and degenerate configurations stay finite (and are
    // then mostly rejected by the barycentric range).
    __host__ __device__ float ray_intersect(const Eigen::Vector3f& ro, const Eigen::Vector3f& rd) const {
        constexpr float MISS = 1e6f;

        const Eigen::Vector3f edge1 = b - a;
        const Eigen::Vector3f edge2 = c - a;

        const Eigen::Vector3f h = rd.cross(edge2);
        const float inv_det = safe_divide(1.0f, edge1.dot(h));

        const Eigen::Vector3f s = ro - a;
        const Eigen::Vector3f q = s.cross(edge1);

        const float u = s.dot(h) * inv_det;
        const float v = rd.dot(q) * inv_det;
        const float t = edge2.dot(q) * inv_det;

        // single-exit acceptance; every comparison fails on NaN
        const bool accept = u >= 0.0f && u <= 1.0f && v >= 0.0f && u + v <= 1.0f && t >= 0.0f;
        return accept ? t : MISS;
    }

    // Closest point to p on the closed segment [s0, s1] (Ericson, Real-Time
    // Collision Detection, ch. 5). The clamp leaves a NaN parameter (from a
    // zero-length segment) untouched so it propagates.
    __host__ __device__ Eigen::Vector3f closest_on_segment(const Eigen::Vector3f& p, const Eigen::Vector3f& s0, const Eigen::Vector3f& s1) const {
        const Eigen::Vector3f d = s1 - s0;
        float t = (p - s0).dot(d) / d.dot(d);
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        return s0 + t * d;
    }

    // Squared distance from p to the triangle: the plane distance when the
    // projection of p falls inside (barycentric test), else the smallest of
    // the three clamped edge-segment distances. A zero-area triangle
    // reports +inf so it can never win a nearest-triangle competition.
    __host__ __device__ float distance_sq(const Eigen::Vector3f& p) const {
        const Eigen::Vector3f e0 = b - a;
        const Eigen::Vector3f e1 = c - a;
        const Eigen::Vector3f w = p - a;

        const float d00 = e0.dot(e0);
        const float d01 = e0.dot(e1);
        const float d11 = e1.dot(e1);
        const float d20 = w.dot(e0);
        const float d11w = w.dot(e1);

        // sign-form barycentric inside test (no divisions)
        const float denom = d00 * d11 - d01 * d01;
        const float num_v = d11 * d20 - d01 * d11w;
        const float num_w = d00 * d11w - d01 * d20;

        if (num_v >= 0.0f && num_w >= 0.0f && num_v + num_w <= denom) {
            // face region (a zero-area triangle lands here with denom == 0
            // and reports +inf, so it never wins)
            const Eigen::Vector3f n = e0.cross(e1);
            const float nn = n.squaredNorm();
            if (nn == 0.0f) {
                return std::numeric_limits<float>::infinity();
            }
            const float h = n.dot(w);
            return h * h / nn;
        }

        // edge regions: clamped segment parameters reuse the dot products
        float t_ab = d20 / d00;
        if (t_ab < 0.0f) t_ab = 0.0f;
        if (t_ab > 1.0f) t_ab = 1.0f;
        const float d_ab = (w - t_ab * e0).squaredNorm();

        float t_ca = (d11 - d11w) / d11;
        if (t_ca < 0.0f) t_ca = 0.0f;
        if (t_ca > 1.0f) t_ca = 1.0f;
        const float d_ca = (w - e1 + t_ca * e1).squaredNorm();

        const Eigen::Vector3f wb = w - e0;
        const Eigen::Vector3f e_bc = e1 - e0;
        float t_bc = wb.dot(e_bc) / e_bc.dot(e_bc);
        if (t_bc < 0.0f) t_bc = 0.0f;
        if (t_bc > 1.0f) t_bc = 1.0f;
        const float d_bc = (wb - t_bc * e_bc).squaredNorm();

        return fminf(fminf(d_ab, d_bc), d_ca);
    }

    // Nearest point on the closed triangle: the plane projection of p when
    // its barycentric coordinates are all non-negative, else the nearest of
    // the three edge segments' closest points. Written so a zero-area
    // triangle falls through to its NaN projection instead of a finite
    // segment point.
    __host__ __device__ Eigen::Vector3f closest_point(Eigen::Vector3f p) const {
        const Eigen::Vector3f n = normal();
        const Eigen::Vector3f on_plane = p - n.dot(p - a) * n;

        const Eigen::Vector3f coords = barycentric(p);
        if (coords.x() < 0.0f || coords.y() < 0.0f || coords.z() < 0.0f) {
            // work from the projection: better conditioned for far-away p,
            // and identical up to the perpendicular offset
            const Eigen::Vector3f q_ab = closest_on_segment(on_plane, a, b);
            const Eigen::Vector3f q_bc = closest_on_segment(on_plane, b, c);
            const Eigen::Vector3f q_ca = closest_on_segment(on_plane, c, a);

            Eigen::Vector3f best = q_ab;
            float best_d = (on_plane - q_ab).squaredNorm();
            const float d_bc = (on_plane - q_bc).squaredNorm();
            if (d_bc < best_d) {
                best = q_bc;
                best_d = d_bc;
            }
            if ((on_plane - q_ca).squaredNorm() < best_d) {
                best = q_ca;
            }
            return best;
        }
        return on_plane;
    }

    __host__ __device__ Eigen::Vector3f barycentric(const Eigen::Vector3f& p) const {
        Eigen::Vector3f v0 = b - a;
        Eigen::Vector3f v1 = c - a;
        Eigen::Vector3f v2 = p - a;

        float d00 = v0.dot(v0);
        float d01 = v0.dot(v1);
        float d11 = v1.dot(v1);
        float d20 = v2.dot(v0);
        float d21 = v2.dot(v1);

        float denom = d00 * d11 - d01 * d01;
        float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom;
        float u = 1.0 - v - w;

        return Eigen::Vector3f(u, v, w);
    }
};

static_assert(sizeof(Triangle) == 48, "Triangle layout is a serialization contract");

} // namespace cubvh
