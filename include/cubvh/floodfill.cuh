#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#define THREADS_PER_BLOCK 256

// Kernel: init labels[idx] = idx for every voxel
__global__ void initLabels(
    const bool* __restrict__ grid,
    int* __restrict__ labels,
    int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= N) return;
    labels[idx] = idx;
}

// Kernel: hook each free voxel toward min‐label of its free 6‐neighbors
__global__ void hook(
    const bool* __restrict__ grid,
          int* __restrict__ labels,
          int* __restrict__ changed,
    int H, int W, int D, int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (grid[idx] != 0) return;  // skip blocked

    int x = idx % W;
    int y = (idx / W) % H;
    int z = idx / (W*H);

    int best = labels[idx];

    // check 6 directions
    if (x > 0) {
        int n = idx - 1;
        if (grid[n]==0) best = min(best, labels[n]);
    }
    if (x < W-1) {
        int n = idx + 1;
        if (grid[n]==0) best = min(best, labels[n]);
    }
    if (y > 0) {
        int n = idx - W;
        if (grid[n]==0) best = min(best, labels[n]);
    }
    if (y < H-1) {
        int n = idx + W;
        if (grid[n]==0) best = min(best, labels[n]);
    }
    if (z > 0) {
        int n = idx - W*H;
        if (grid[n]==0) best = min(best, labels[n]);
    }
    if (z < D-1) {
        int n = idx + W*H;
        if (grid[n]==0) best = min(best, labels[n]);
    }

    if (best < labels[idx]) {
        labels[idx] = best;
        atomicOr(changed, 1);
    }
}

// Kernel: one pointer-jumping compress step
__global__ void compress(
    int* __restrict__ labels,
    int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int p = labels[idx];
    labels[idx] = labels[p];
}

extern "C"
void _floodfill(
    const bool *grid, 
    const int H, const int W, const int D,
    int32_t *mask)    // now int32_t per‐voxel
{
    const int N = H * W * D;
    const size_t bytesGrid  = N * sizeof(bool);
    const size_t bytesLabel = N * sizeof(int);

    // Device buffers
    bool *d_grid = nullptr;
    int *d_labels = nullptr, *d_changed = nullptr;

    // Allocate
    cudaMalloc(&d_grid,    bytesGrid);
    cudaMalloc(&d_labels,  bytesLabel);
    cudaMalloc(&d_changed, sizeof(int));

    // Copy grid
    cudaMemcpy(d_grid, grid, bytesGrid, cudaMemcpyHostToDevice);

    // initLabels
    {
        int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        initLabels<<<blocks, THREADS_PER_BLOCK>>>(d_grid, d_labels, N);
    }

    // iterate hook+compress until no change
    int h_changed = 0;
    do {
        cudaMemset(d_changed, 0, sizeof(int));

        // hook
        {
            int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            hook<<<blocks, THREADS_PER_BLOCK>>>(
                d_grid, d_labels, d_changed,
                H, W, D, N
            );
        }
        // compress
        {
            int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            compress<<<blocks, THREADS_PER_BLOCK>>>(d_labels, N);
        }
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    } while (h_changed);

    // final compress to fully flatten
    {
        int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        compress<<<blocks, THREADS_PER_BLOCK>>>(d_labels, N);
        cudaDeviceSynchronize();
    }

    // copy labels → mask (int32)
    cudaMemcpy(mask, d_labels, bytesLabel, cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(d_grid);
    cudaFree(d_labels);
    cudaFree(d_changed);
}