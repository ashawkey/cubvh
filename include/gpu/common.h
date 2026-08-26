#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <Eigen/Dense>

namespace cubvh {

constexpr float PI = 3.14159265358979323846f;
constexpr float SQRT2 = 1.41421356237309504880f;

template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

constexpr uint32_t N_THREADS_LINEAR = 128;

#if defined(__CUDACC__) || defined(__HIPCC__)
// Launch a 1-D grid, one thread per element, 128 threads per block.
// The kernel receives the element count as its first argument.
template <typename K, typename T, typename... Types>
inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types... args) {
    if (n_elements > 0) {
        const uint32_t n = (uint32_t)n_elements;
        kernel<<<div_round_up(n, N_THREADS_LINEAR), N_THREADS_LINEAR, shmem_size, stream>>>(n, args...);
    }
}
#endif

inline __host__ __device__ float sign(float x) {
    return copysignf(1.0f, x);
}

inline __host__ __device__ float fractf(float x) {
    return x - floorf(x);
}

inline __host__ __device__ float safe_divide(float numerator, float denominator, float epsilon = 1e-6f) {
	if (fabs(denominator) < epsilon) {
		if (denominator <= 0)
			return -(numerator / epsilon);
		else
			return numerator / epsilon;
	}
	return numerator / denominator;
}

// Lattice offset by set size, per the table in R. Roberts, "How to evenly
// distribute points on a sphere more effectively than the canonical
// Fibonacci lattice".
inline __host__ __device__ float fibonacci_lattice_epsilon(uint32_t n) {
    if (n >= 11000) return 27.0f;
    if (n >= 890) return 10.0f;
    if (n >= 177) return 3.33f;
    if (n >= 24) return 1.33f;
    return 0.33f;
}

// i-th direction of an offset Fibonacci lattice mapped to the unit sphere.
// `offset` shifts the lattice in the unit square (randomized rotation).
template <uint32_t N_DIRS>
__host__ __device__ Eigen::Vector3f fibonacci_dir(uint32_t i, const Eigen::Vector2f& offset) {
    constexpr float golden = 1.6180339887498948482045868343656f;
    const float epsilon = fibonacci_lattice_epsilon(N_DIRS);

    const float u = fractf((i + epsilon) / (N_DIRS - 1 + 2 * epsilon) + offset.x());
    const float v = fractf(i / golden + offset.y());

    // cylindrical equal-area map of the unit square to the sphere
    const float cos_theta = 1.0f - 2.0f * u;
    const float sin_theta_sq = fmaxf(1.0f - cos_theta * cos_theta, 0.0f);
    const float sin_theta = sqrtf(sin_theta_sq);
    float sin_phi, cos_phi;
    sincosf(2.0f * PI * (v - 0.5f), &sin_phi, &cos_phi);

    return Eigen::Vector3f(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
}

} // namespace cubvh
