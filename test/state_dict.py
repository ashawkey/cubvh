import numpy as np
import trimesh
import argparse
import torch
import cubvh
import tempfile


def create_dodecahedron(radius=1, center=np.array([0, 0, 0])):

    vertices = np.array([
        -0.57735,  -0.57735,  0.57735,
        0.934172,  0.356822,  0,
        0.934172,  -0.356822,  0,
        -0.934172,  0.356822,  0,
        -0.934172,  -0.356822,  0,
        0,  0.934172,  0.356822,
        0,  0.934172,  -0.356822,
        0.356822,  0,  -0.934172,
        -0.356822,  0,  -0.934172,
        0,  -0.934172,  -0.356822,
        0,  -0.934172,  0.356822,
        0.356822,  0,  0.934172,
        -0.356822,  0,  0.934172,
        0.57735,  0.57735,  -0.57735,
        0.57735,  0.57735,  0.57735,
        -0.57735,  0.57735,  -0.57735,
        -0.57735,  0.57735,  0.57735,
        0.57735,  -0.57735,  -0.57735,
        0.57735,  -0.57735,  0.57735,
        -0.57735,  -0.57735,  -0.57735,
        ]).reshape((-1,3), order="C")

    faces = np.array([
        19, 3, 2,
        12, 19, 2,
        15, 12, 2,
        8, 14, 2,
        18, 8, 2,
        3, 18, 2,
        20, 5, 4,
        9, 20, 4,
        16, 9, 4,
        13, 17, 4,
        1, 13, 4,
        5, 1, 4,
        7, 16, 4,
        6, 7, 4,
        17, 6, 4,
        6, 15, 2,
        7, 6, 2,
        14, 7, 2,
        10, 18, 3,
        11, 10, 3,
        19, 11, 3,
        11, 1, 5,
        10, 11, 5,
        20, 10, 5,
        20, 9, 8,
        10, 20, 8,
        18, 10, 8,
        9, 16, 7,
        8, 9, 7,
        14, 8, 7,
        12, 15, 6,
        13, 12, 6,
        17, 13, 6,
        13, 1, 11,
        12, 13, 11,
        19, 12, 11,
        ]).reshape((-1, 3), order="C")-1

    length = np.linalg.norm(vertices, axis=1).reshape((-1, 1))
    vertices = vertices / length * radius + center

    return trimesh.Trimesh(vertices=vertices, faces=faces)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', default=1000, type=int)
    parser.add_argument('--mesh', default='', type=str)
    
    opt = parser.parse_args()

    if opt.mesh == '':
        mesh = create_dodecahedron()
    else:
        mesh = trimesh.load(opt.mesh, force='mesh', skip_material=True)


    # query nearest triangles for a set of points
    points = torch.randn(opt.N, 3, device='cuda', dtype=torch.float32)

    # Initialize BVH
    BVH = cubvh.cuBVH(mesh.vertices, mesh.faces)
    distances, face_id, uvw = BVH.unsigned_distance(points, return_uvw=True)
    
    # Save state_dict to disk
    state_dict = BVH.state_dict()
    with tempfile.NamedTemporaryFile() as f:
        torch.save(state_dict, f.name)

        # Load state_dict from disk
        loaded_state_dict = torch.load(f.name)
    BVH_loaded = cubvh.cuBVH.from_state_dict(loaded_state_dict)
        
    # Verify that the loaded BVH gives the same results
    distances_loaded, face_id_loaded, uvw_loaded = BVH_loaded.unsigned_distance(points, return_uvw=True)
    assert torch.allclose(distances, distances_loaded)
    assert torch.all(face_id == face_id_loaded)
    assert torch.allclose(uvw, uvw_loaded)
    print("State dict save/load test passed.")
    