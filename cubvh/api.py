import numpy as np
import torch

# CUDA extension
import _cubvh as _backend

_sdf_mode_to_id = {
    'watertight': 0,
    'raystab': 1,
}

class cuBVH():
    def __init__(self, vertices, triangles):
        # vertices: np.ndarray, [N, 3]
        # triangles: np.ndarray, [M, 3]

        if torch.is_tensor(vertices): vertices = vertices.detach().cpu().numpy()
        if torch.is_tensor(triangles): triangles = triangles.detach().cpu().numpy()

        # check inputs
        assert triangles.shape[0] > 8, "BVH needs at least 8 triangles."
        
        # implementation
        self.impl = _backend.create_cuBVH(vertices, triangles)

    def ray_trace(self, rays_o, rays_d):
        # rays_o: torch.Tensor, float, [N, 3]
        # rays_d: torch.Tensor, float, [N, 3]

        rays_o = rays_o.float().contiguous()
        rays_d = rays_d.float().contiguous()

        if not rays_o.is_cuda: rays_o = rays_o.cuda()
        if not rays_d.is_cuda: rays_d = rays_d.cuda()

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.view(-1, 3)
        rays_d = rays_d.view(-1, 3)

        N = rays_o.shape[0]

        # init output buffers
        positions = torch.empty(N, 3, dtype=torch.float32, device=rays_o.device)
        face_id = torch.empty(N, dtype=torch.int64, device=rays_o.device)
        depth = torch.empty(N, dtype=torch.float32, device=rays_o.device)
        
        self.impl.ray_trace(rays_o, rays_d, positions, face_id, depth) # [N, 3]

        positions = positions.view(*prefix, 3)
        face_id = face_id.view(*prefix)
        depth = depth.view(*prefix)

        return positions, face_id, depth

    def unsigned_distance(self, positions, return_uvw=False):
        # positions: torch.Tensor, float, [N, 3]

        positions = positions.float().contiguous()

        if not positions.is_cuda: positions = positions.cuda()

        prefix = positions.shape[:-1]
        positions = positions.view(-1, 3)

        N = positions.shape[0]

        # init output buffers
        distances = torch.empty(N, dtype=torch.float32, device=positions.device)
        face_id = torch.empty(N, dtype=torch.int64, device=positions.device)

        if return_uvw:
            uvw = torch.empty(N, 3, dtype=torch.float32, device=positions.device)
        else:
            uvw = None
        
        self.impl.unsigned_distance(positions, distances, face_id, uvw) # [N, 3]

        distances = distances.view(*prefix)
        face_id = face_id.view(*prefix)
        if uvw is not None:
            uvw = uvw.view(*prefix, 3)

        return distances, face_id, uvw

    
    def signed_distance(self, positions, return_uvw=False, mode='watertight'):
        # positions: torch.Tensor, float, [N, 3]

        positions = positions.float().contiguous()

        if not positions.is_cuda: positions = positions.cuda()

        prefix = positions.shape[:-1]
        positions = positions.view(-1, 3)

        N = positions.shape[0]

        # init output buffers
        distances = torch.empty(N, dtype=torch.float32, device=positions.device)
        face_id = torch.empty(N, dtype=torch.int64, device=positions.device)

        if return_uvw:
            uvw = torch.empty(N, 3, dtype=torch.float32, device=positions.device)
        else:
            uvw = None
        
        self.impl.signed_distance(positions, distances, face_id, uvw, _sdf_mode_to_id[mode]) # [N, 3]

        distances = distances.view(*prefix)
        face_id = face_id.view(*prefix)
        if uvw is not None:
            uvw = uvw.view(*prefix, 3)

        return distances, face_id, uvw

def floodfill(grid):
    # grid: torch.Tensor, uint8, [B, H, W, D] or [H, W, D]
    # return: torch.Tensor, int32, [B, H, W, D] or [H, W, D], label of the connected component (value can be 0 to H*W*D-1, not remapped!)

    grid = grid.contiguous()
    if not grid.is_cuda: grid = grid.cuda()

    if grid.dim() == 3:
        mask = _backend.floodfill(grid.unsqueeze(0)).squeeze(0)
    else:
        mask = _backend.floodfill(grid)

    return mask

def sparse_marching_cubes(coords, corners, iso, ensure_consistency=False):
    # coords: torch.Tensor, int32, [N, 3]
    # corners: torch.Tensor, float32, [N, 8]
    # iso: float
    # ensure_consistency: bool, whether to ensure shared corner values are consistent

    coords = coords.int().contiguous()
    corners = corners.float().contiguous()

    if not coords.is_cuda: coords = coords.cuda()
    if not corners.is_cuda: corners = corners.cuda()

    verts, tris = _backend.sparse_marching_cubes(coords, corners, iso, ensure_consistency)

    return verts, tris

# CPU hole filling numpy API
def fill_holes(vertices: np.ndarray, faces: np.ndarray, return_added: bool = False, check_containment: bool = True, eps: float = 1e-7, verbose: bool = False) -> np.ndarray:
    """
    Fill small holes in a triangular mesh using a CPU ear-clipping strategy.

    Args:
        vertices (np.ndarray float32 [N,3])
        faces (np.ndarray int32 [M,3])
        return_added: if True, return only newly added triangles; else full face list
        check_containment: avoid creating triangles containing other boundary verts
        eps: numeric epsilon
        verbose: print detailed logs from C++
    Returns:
        np.ndarray int32 [...,3]
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    return _backend.fill_holes(vertices, faces, return_added, check_containment, float(eps), bool(verbose))

def merge_vertices(vertices: np.ndarray, faces: np.ndarray, threshold: float = 1e-3):
    """Merge vertices closer than threshold.
    Args:
        vertices (np.ndarray float32 [N,3])
        faces (np.ndarray int32 [M,3])
        threshold (float): distance threshold
    Returns:
        (vertices, faces) after merging
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    assert vertices.ndim==2 and vertices.shape[1]==3
    assert faces.ndim==2 and faces.shape[1]==3
    v_new, f_new = _backend.merge_vertices(vertices, faces, float(threshold))
    return np.asarray(v_new, dtype=np.float32), np.asarray(f_new, dtype=np.int32)