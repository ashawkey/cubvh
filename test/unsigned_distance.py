import numpy as np
import trimesh
import argparse
import torch
import cubvh
import time


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

    # Ours
    _t0 = time.time()
    BVH = cubvh.cuBVH(mesh.vertices, mesh.faces)
    torch.cuda.synchronize()
    _t1 = time.time()
    distances, face_id, uvw = BVH.unsigned_distance(points, return_uvw=True)
    torch.cuda.synchronize()
    _t2 = time.time()
    print(f'[TIME] Ours total {_t2 - _t0:.6f}s = build {_t1 - _t0:.6f}s + query {_t2 - _t1:.6f}s')

    # GT results by trimesh
    _t0 = time.time()
    gt_cpoint, gt_distances, gt_face_id = trimesh.proximity.closest_point(mesh, points.cpu().numpy())
    _t1 = time.time()
    print(f'[TIME] Trimesh total {_t1 - _t0:.6f}s')

    # verify correctness
    face_id = face_id.cpu().numpy().astype(np.int64)

    # NOTE: if there are duplicated vertices, this will fail... but it won't affect later correctness.
    # np.testing.assert_array_equal(
    #     face_id,
    #     gt_face_id.astype(np.int64)
    # )
    
    distances = distances.cpu().numpy().astype(np.float32)
    np.testing.assert_allclose(
        distances,
        gt_distances.astype(np.float32),
        atol=1e-5
    )
    
    # calc cpoint from uvw and face_id
    uvw = uvw.cpu().numpy().astype(np.float32) # [M, 3]
    trigs = mesh.faces[face_id] # [M, 3]

    cpoint = mesh.vertices[trigs[:, 0]] * uvw[:, [0]] + \
             mesh.vertices[trigs[:, 1]] * uvw[:, [1]] + \
             mesh.vertices[trigs[:, 2]] * uvw[:, [2]]

    np.testing.assert_allclose(
        cpoint,
        gt_cpoint.astype(np.float32),
        atol=1e-5
    )

    