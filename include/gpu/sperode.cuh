// Sparse erosion on a binary 3D volume represented by occupied voxel coords.
// A voxel remains iff all its 6-neighbours also exist (cross structuring element).

#include <cuda.h>  
#include <cuda_runtime.h>

#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/transform_iterator.h>
#include <climits>

#define THREADS_PER_BLOCK 256

// Lexicographic comparator for int3
struct Int3Less {
    __host__ __device__ bool operator()(const int3 &a, const int3 &b) const {
        if (a.x < b.x) return true; if (a.x > b.x) return false;
        if (a.y < b.y) return true; if (a.y > b.y) return false;
        return a.z < b.z;
    }
};

// Binary search for int3 within a sorted array (lexicographic order)
__device__ __forceinline__ bool contains_int3(const int3 *arr, int n, const int3 &key) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int3 v = arr[mid];
        if (v.x == key.x && v.y == key.y && v.z == key.z) return true;
        // lexicographic compare
        if (v.x < key.x || (v.x == key.x && (v.y < key.y || (v.y == key.y && v.z < key.z)))) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return false;
}

// Pack raw coords [N*3] into int3 array
__global__ void pack_coords_kernel(const int *coords, int3 *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int x = coords[3 * i + 0];
    int y = coords[3 * i + 1];
    int z = coords[3 * i + 2];
    out[i] = make_int3(x, y, z);
}

// Erosion check kernel: mask[i] = true if all 6-neighbours exist, else false.
__global__ void erode_check_kernel(const int3 *coords, const int3 *sorted, int N, bool *mask) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const int3 c = coords[i];
    // 6-neighbour offsets
    int3 nbs[6];
    nbs[0] = make_int3(c.x - 1, c.y, c.z);
    nbs[1] = make_int3(c.x + 1, c.y, c.z);
    nbs[2] = make_int3(c.x, c.y - 1, c.z);
    nbs[3] = make_int3(c.x, c.y + 1, c.z);
    nbs[4] = make_int3(c.x, c.y, c.z - 1);
    nbs[5] = make_int3(c.x, c.y, c.z + 1);

    bool keep = true;
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        if (!contains_int3(sorted, N, nbs[k])) { keep = false; break; }
    }
    mask[i] = keep;
}

// ----- Bitset path ---------------------------------------------------------

// Set bits for occupied voxels in dense 1-bit volume
__global__ void set_bits_kernel(const int3 *coords, int N, int3 mn, int dimX, int dimY, int dimZ, unsigned int *bits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int3 c = coords[i];
    int lx = c.x - mn.x;
    int ly = c.y - mn.y;
    int lz = c.z - mn.z;
    if (lx < 0 || lx >= dimX || ly < 0 || ly >= dimY || lz < 0 || lz >= dimZ) return; // safety
    unsigned long long lin = (static_cast<unsigned long long>(lz) * dimY + ly) * dimX + lx;
    unsigned int word = static_cast<unsigned int>(lin >> 5ULL);
    unsigned int bit  = static_cast<unsigned int>(lin & 31ULL);
    atomicOr(bits + word, 1u << bit);
}

__device__ __forceinline__ bool bit_test(const unsigned int *bits, unsigned long long lin) {
    unsigned int word = static_cast<unsigned int>(lin >> 5ULL);
    unsigned int bit  = static_cast<unsigned int>(lin & 31ULL);
    return (bits[word] >> bit) & 1u;
}

__global__ void erode_check_bitset_kernel(const int3 *coords, int N, int3 mn, int dimX, int dimY, int dimZ, const unsigned int *bits, bool *mask) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int3 c = coords[i];
    int lx = c.x - mn.x;
    int ly = c.y - mn.y;
    int lz = c.z - mn.z;
    bool keep = true;
    // neighbours relative positions
    const int3 offs[6] = { make_int3(-1,0,0), make_int3(1,0,0), make_int3(0,-1,0), make_int3(0,1,0), make_int3(0,0,-1), make_int3(0,0,1) };
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        int nx = lx + offs[k].x;
        int ny = ly + offs[k].y;
        int nz = lz + offs[k].z;
        if (nx < 0 || nx >= dimX || ny < 0 || ny >= dimY || nz < 0 || nz >= dimZ) { keep = false; break; }
        unsigned long long lin = (static_cast<unsigned long long>(nz) * dimY + ny) * dimX + nx;
        if (!bit_test(bits, lin)) { keep = false; break; }
    }
    mask[i] = keep;
}

// Functors to extract components for thrust transform_iterator
struct GetX { __host__ __device__ int operator()(const int3 &a) const { return a.x; } };
struct GetY { __host__ __device__ int operator()(const int3 &a) const { return a.y; } };
struct GetZ { __host__ __device__ int operator()(const int3 &a) const { return a.z; } };

// coords: [N * 3] (device pointer), mask: [N] (device pointer)
// Erodes by one voxel using 6-neighbour cross.
inline void _sparse_erode(const int *coords, const int N, bool *mask, cudaStream_t stream = 0) {  
    // Pack coords into int3 device array
    thrust::device_vector<int3> d_coords(N);
    const int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    pack_coords_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(coords, thrust::raw_pointer_cast(d_coords.data()), N);

    // Compute bounding box on device
    auto begin = d_coords.begin();
    auto bx = thrust::make_transform_iterator(begin, GetX());
    auto by = thrust::make_transform_iterator(begin, GetY());
    auto bz = thrust::make_transform_iterator(begin, GetZ());
    auto ex = bx + N;
    auto ey = by + N;
    auto ez = bz + N;

    // Compute mins/maxs via device reductions and return to host
    int h_minx = thrust::reduce(thrust::cuda::par.on(stream), bx, ex, INT_MAX, thrust::minimum<int>());
    int h_maxx = thrust::reduce(thrust::cuda::par.on(stream), bx, ex, INT_MIN, thrust::maximum<int>());
    int h_miny = thrust::reduce(thrust::cuda::par.on(stream), by, ey, INT_MAX, thrust::minimum<int>());
    int h_maxy = thrust::reduce(thrust::cuda::par.on(stream), by, ey, INT_MIN, thrust::maximum<int>());
    int h_minz = thrust::reduce(thrust::cuda::par.on(stream), bz, ez, INT_MAX, thrust::minimum<int>());
    int h_maxz = thrust::reduce(thrust::cuda::par.on(stream), bz, ez, INT_MIN, thrust::maximum<int>());

    int dimX = h_maxx - h_minx + 1;
    int dimY = h_maxy - h_miny + 1;
    int dimZ = h_maxz - h_minz + 1;

    // Guard against empty input
    if (N == 0) return;

    // Decide whether to use bitset path (limit memory to 256 MB)
    const unsigned long long voxels = static_cast<unsigned long long>(dimX) * dimY * dimZ;
    const unsigned long long bits_needed = voxels;
    const unsigned long long bytes_needed = (bits_needed + 7ULL) / 8ULL;
    const unsigned long long MAX_BYTES = 256ULL * 1024ULL * 1024ULL; // 256 MB

    if (bytes_needed > 0 && bytes_needed <= MAX_BYTES) {
        // Build dense bitset
        const unsigned long long words = (bits_needed + 31ULL) / 32ULL;
        thrust::device_vector<unsigned int> d_bits(words);
        cudaMemsetAsync(thrust::raw_pointer_cast(d_bits.data()), 0, words * sizeof(unsigned int), stream);

        int3 mn = make_int3(h_minx, h_miny, h_minz);
        set_bits_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            thrust::raw_pointer_cast(d_coords.data()), N, mn, dimX, dimY, dimZ,
            thrust::raw_pointer_cast(d_bits.data())
        );

        erode_check_bitset_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            thrust::raw_pointer_cast(d_coords.data()), N, mn, dimX, dimY, dimZ,
            thrust::raw_pointer_cast(d_bits.data()), mask
        );
    } else {
        // Fallback to sort + binary search membership checks
        thrust::device_vector<int3> d_sorted = d_coords;
        thrust::sort(thrust::cuda::par.on(stream), d_sorted.begin(), d_sorted.end(), Int3Less{});

        erode_check_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            thrust::raw_pointer_cast(d_coords.data()),
            thrust::raw_pointer_cast(d_sorted.data()),
            N,
            mask);
    }
}  
