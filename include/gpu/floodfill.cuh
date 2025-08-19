#include <cuda.h>  
#include <cuda_runtime.h>

#include <cstdint>

#define THREADS_PER_BLOCK 256

/* ------------------------------------------------------------------------- */  
/*  K E R N E L S                                                            */  
/* ------------------------------------------------------------------------- */

// Initialise: label[i] = i  
__global__ void initLabels(int * __restrict__ labels, int Ntot)  
{  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (idx < Ntot) labels[idx] = idx;  
}

/*  
 * Hook step for batched volumes.  
 *  - Nvol  := H * W * D            (size of ONE volume)  
 *  - Ntot  := B * Nvol             (size of the whole batch)  
 *  
 * The only difference to the single-volume version is that we build the  
 * voxel coordinates (x,y,z) from the *local* index inside the volume  
 * instead of the global thread index.  
 */  
__global__ void hookBatch(  
    const bool* __restrict__ grid,  
          int*  __restrict__ labels,  
          int*  __restrict__ changed,  
    int H, int W, int D,  
    int Nvol,                // stride between consecutive batches  
    int Ntot)                // total number of voxels  
{  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (idx >= Ntot) return;  
    if (grid[idx] != 0) return;        // blocked voxel

    /* ---------- local coordinates inside the current volume -------------- */  
    int local = idx % Nvol;            // 0 … Nvol-1  
    int x =  local % W;  
    int y = (local / W) % H;  
    int z =  local / (W * H);

    /* ----------------------- six-neighbour search ------------------------ */  
    int best = labels[idx];

    if (x > 0)     { int n = idx - 1;      if (!grid[n]) best = min(best, labels[n]); }  
    if (x < W - 1) { int n = idx + 1;      if (!grid[n]) best = min(best, labels[n]); }

    if (y > 0)     { int n = idx - W;      if (!grid[n]) best = min(best, labels[n]); }  
    if (y < H - 1) { int n = idx + W;      if (!grid[n]) best = min(best, labels[n]); }

    if (z > 0)     { int n = idx - W * H;  if (!grid[n]) best = min(best, labels[n]); }  
    if (z < D - 1) { int n = idx + W * H;  if (!grid[n]) best = min(best, labels[n]); }

    /* --------------------------- update ---------------------------------- */  
    if (best < labels[idx]) {  
        labels[idx] = best;  
        atomicOr(changed, 1);  
    }  
}

/* One pointer-jump compression step (unchanged) */  
__global__ void compress(int * __restrict__ labels, int Ntot)  
{  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  
    if (idx >= Ntot) return;  
    int p = labels[idx];  
    labels[idx] = labels[p];  
}

/* ------------------------------------------------------------------------- */  
/*  H O S T   U T I L I T Y                                                  */  
/* ------------------------------------------------------------------------- */

static int divUp(int a, int b) { return (a + b - 1) / b; }

/* ------------------------------------------------------------------------- */  
/*  P U B L I C   A P I                                                      */  
/* ------------------------------------------------------------------------- */

/*  
 * grid  : pointer to B·H·W·D bools (0 = free, 1 = blocked)  
 * mask  : output, B·H·W·D int32 labels  
 */  
extern "C"  
void _floodfill_batch(  
        const bool *grid,  
        int B, int H, int W, int D,  
        int32_t *mask)  
{  
    const int Nvol = H * W * D;          // size of one volume  
    const int Ntot = B * Nvol;           // size of entire batch

    const size_t bytesGrid  = Ntot * sizeof(bool);  
    const size_t bytesLabel = Ntot * sizeof(int);

    /* --------------------------- allocate -------------------------------- */  
    bool *d_grid    = nullptr;  
    int  *d_labels  = nullptr;  
    int  *d_changed = nullptr;

    cudaMalloc(&d_grid,    bytesGrid);  
    cudaMalloc(&d_labels,  bytesLabel);  
    cudaMalloc(&d_changed, sizeof(int));

    cudaMemcpy(d_grid, grid, bytesGrid, cudaMemcpyHostToDevice);

    /* --------------------------- init ------------------------------------ */  
    {  
        int blocks = divUp(Ntot, THREADS_PER_BLOCK);  
        initLabels<<<blocks, THREADS_PER_BLOCK>>>(d_labels, Ntot);  
    }

    /* -------------------- iterated hook / compress ----------------------- */  
    int h_changed;  
    do {  
        cudaMemset(d_changed, 0, sizeof(int));

        /* hook */  
        {  
            int blocks = divUp(Ntot, THREADS_PER_BLOCK);  
            hookBatch<<<blocks, THREADS_PER_BLOCK>>>(  
                d_grid, d_labels, d_changed,  
                H, W, D, Nvol, Ntot);  
        }

        /* compress */  
        {  
            int blocks = divUp(Ntot, THREADS_PER_BLOCK);  
            compress<<<blocks, THREADS_PER_BLOCK>>>(d_labels, Ntot);  
        }

        cudaDeviceSynchronize();  
        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);  
    } while (h_changed);

    /* final flatten */  
    {  
        int blocks = divUp(Ntot, THREADS_PER_BLOCK);  
        compress<<<blocks, THREADS_PER_BLOCK>>>(d_labels, Ntot);  
        cudaDeviceSynchronize();  
    }

    /* copy back */  
    cudaMemcpy(mask, d_labels, bytesLabel, cudaMemcpyDeviceToHost);

    /* cleanup */  
    cudaFree(d_grid);  
    cudaFree(d_labels);  
    cudaFree(d_changed);  
}  
