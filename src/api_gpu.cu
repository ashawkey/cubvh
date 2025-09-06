#include <gpu/api_gpu.h>
#include <gpu/common.h>
#include <gpu/bvh.cuh>
#include <gpu/floodfill.cuh>
#include <gpu/spcumc.cuh>
#include <gpu/hashtable.cuh>

#include <Eigen/Dense>

using namespace Eigen;

using Verts = Matrix<float, Dynamic, 3, RowMajor>;
using Trigs = Matrix<uint32_t, Dynamic, 3, RowMajor>;

namespace cubvh {

class cuBVHImpl : public cuBVH {
public:

    // accept numpy array (cpu) to init 
    cuBVHImpl(Ref<const Verts> vertices, Ref<const Trigs> triangles) : cuBVH() {

        const size_t n_vertices = vertices.rows();
        const size_t n_triangles = triangles.rows();

        triangles_cpu.resize(n_triangles);

        for (size_t i = 0; i < n_triangles; i++) {
            triangles_cpu[i] = {vertices.row(triangles(i, 0)), vertices.row(triangles(i, 1)), vertices.row(triangles(i, 2)), (int64_t)i};
        }

        if (!triangle_bvh) {
            triangle_bvh = TriangleBvh::make();
        }

        triangle_bvh->build(triangles_cpu, 8);

        triangles_gpu.resize_and_copy_from_host(triangles_cpu);

        // TODO: need OPTIX
        // triangle_bvh->build_optix(triangles_gpu, m_inference_stream);

    }

    void ray_trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor positions, at::Tensor face_id, at::Tensor depth) {

        const uint32_t n_elements = rays_o.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        triangle_bvh->ray_trace_gpu(n_elements, rays_o.data_ptr<float>(), rays_d.data_ptr<float>(), positions.data_ptr<float>(), face_id.data_ptr<int64_t>(), depth.data_ptr<float>(), triangles_gpu.data(), stream);
    }

    void unsigned_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw) {

        const uint32_t n_elements = positions.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        triangle_bvh->unsigned_distance_gpu(n_elements, positions.data_ptr<float>(), distances.data_ptr<float>(), face_id.data_ptr<int64_t>(), uvw.has_value() ? uvw.value().data_ptr<float>() : nullptr, triangles_gpu.data(), stream);

    }

    void signed_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw, uint32_t mode) {

        const uint32_t n_elements = positions.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        triangle_bvh->signed_distance_gpu(n_elements, mode, positions.data_ptr<float>(), distances.data_ptr<float>(), face_id.data_ptr<int64_t>(), uvw.has_value() ? uvw.value().data_ptr<float>() : nullptr, triangles_gpu.data(), stream);
    }

    std::vector<Triangle> triangles_cpu;
    GPUMemory<Triangle> triangles_gpu;
    std::shared_ptr<TriangleBvh> triangle_bvh;
};
    
cuBVH* create_cuBVH(Ref<const Verts> vertices, Ref<const Trigs> triangles) {
    return new cuBVHImpl{vertices, triangles};
}

at::Tensor floodfill(at::Tensor grid) {

    // assert grid is uint8_t
    assert(grid.dtype() == at::ScalarType::Bool);

    const int B = grid.size(0);
    const int H = grid.size(1);
    const int W = grid.size(2);
    const int D = grid.size(3);

    // allocate mask
    at::Tensor mask = at::zeros({B, H, W, D}, at::device(grid.device()).dtype(at::ScalarType::Int));

    _floodfill_batch(grid.data_ptr<bool>(), B, H, W, D, mask.data_ptr<int32_t>());

    return mask;
}

std::tuple<at::Tensor, at::Tensor> sparse_marching_cubes(
    at::Tensor coords,        // [N,3] int32, cuda
    at::Tensor corners,       // [N,8] float32, cuda
    double iso_d,             // (PyTorch passes double ⇒ cast to float)
    bool ensure_consistency)  // whether to ensure corner consistency
{
    TORCH_CHECK(coords.is_cuda(),  "coords must reside on CUDA");
    TORCH_CHECK(corners.is_cuda(), "corners must reside on CUDA");
    TORCH_CHECK(coords.dtype()  == at::kInt,   "coords must be int32");
    TORCH_CHECK(corners.dtype() == at::kFloat, "corners must be float32");
    TORCH_CHECK(coords.sizes().size()  == 2 && coords.size(1)  == 3,
                "coords must be of shape [N,3]");
    TORCH_CHECK(corners.sizes().size() == 2 && corners.size(1) == 8,
                "corners must be of shape [N,8]");
    TORCH_CHECK(coords.size(0) == corners.size(0),
                "coords and corners must have the same first-dim (N)");

    // Ensure contiguous memory - PyTorch extensions expect this.
    coords  = coords.contiguous();
    corners = corners.contiguous();
    const int    N   = static_cast<int>(coords.size(0));
    const int   *d_coords  = coords.data_ptr<int>();
    const float *d_corners = corners.data_ptr<float>();
    const float  iso       = static_cast<float>(iso_d);

    // Use the current PyTorch CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // --- call the CUDA sparse MC core (header we wrote earlier) -------------------
    auto mesh = _sparse_marching_cubes(d_coords, d_corners, N, iso, ensure_consistency, stream);
    thrust::device_vector<V3f> &verts_vec = mesh.first;
    thrust::device_vector<Tri> &tris_vec  = mesh.second;
    const int64_t M = static_cast<int64_t>(verts_vec.size());
    const int64_t T = static_cast<int64_t>(tris_vec.size());

    // --- create output tensors ----------------------------------------------------
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(coords.device());
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(coords.device());

    at::Tensor verts = at::empty({M, 3}, opts_f);
    at::Tensor tris  = at::empty({T, 3}, opts_i);

    // Copy GPU→GPU (same stream ⇒ async & cheap)
    cudaMemcpyAsync(verts.data_ptr<float>(),
                    thrust::raw_pointer_cast(verts_vec.data()),
                    M * 3 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    cudaMemcpyAsync(tris.data_ptr<int>(),
                    thrust::raw_pointer_cast(tris_vec.data()),
                    T * 3 * sizeof(int),
                    cudaMemcpyDeviceToDevice, stream);

    // Make sure copies finish before we free device_vectors
    cudaStreamSynchronize(stream);

    return {verts, tris};
}

// ------------------------ GPU Hash Table bindings (virtual pattern) ----------

class cuHashTableImpl : public cuHashTable {
public:
    cuHashTableImpl() {}
    ~cuHashTableImpl() override {}

    void set_num_dims(int d) override {
        ht.set_num_dims(d);
    }

    int get_num_dims() const override {
        return ht.num_dims;
    }

    void resize(int capacity) override {
        ht.resize(capacity);
    }

    void prepare() override {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        ht.prepare(stream);
    }

    void insert(at::Tensor coords) override {
        TORCH_CHECK(coords.is_cuda(),  "coords must reside on CUDA");
        TORCH_CHECK(coords.dtype()  == at::kInt,   "coords must be int32");
        TORCH_CHECK(coords.dim() == 2, "coords must be 2D [N,D]");
        coords = coords.contiguous();
        const int N = (int)coords.size(0);
        const int D = (int)coords.size(1);
        ht.set_num_dims(D);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        ht.insert(coords.data_ptr<int>(), N, stream);
    }

    void build(at::Tensor coords) override {
        TORCH_CHECK(coords.is_cuda(),  "coords must reside on CUDA");
        TORCH_CHECK(coords.dtype()  == at::kInt,   "coords must be int32");
        TORCH_CHECK(coords.dim() == 2, "coords must be 2D [N,D]");
        coords = coords.contiguous();
        const int N = (int)coords.size(0);
        const int D = (int)coords.size(1);
        ht.set_num_dims(D);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        ht.build(coords.data_ptr<int>(), N, stream);
    }

    at::Tensor search(at::Tensor queries) const override {
        TORCH_CHECK(queries.is_cuda(),  "queries must reside on CUDA");
        TORCH_CHECK(queries.dtype()  == at::kInt,   "queries must be int32");
        TORCH_CHECK(queries.dim() == 2, "queries must be 2D [M,D]");
        TORCH_CHECK(ht.capacity > 0, "hash table is not built");
        at::Tensor q = queries.contiguous();
        const int M = (int)q.size(0);
        auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(q.device());
        at::Tensor out = at::empty({M}, opts_i);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        ht.search(q.data_ptr<int>(), M, out.data_ptr<int>(), stream);
        return out;
    }

private:
    HashTableInt ht;
};

cuHashTable* create_cuHashTable() {
    return new cuHashTableImpl{};
}

} // namespace cubvh