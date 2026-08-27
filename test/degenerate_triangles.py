import numpy as np
import torch

import cubvh


def check_no_surface_result(result):
    distances, face_id, uvw = result
    assert torch.isposinf(distances).all()
    assert (face_id == -1).all()
    assert (uvw == 0).all()


def main():
    vertices = np.zeros((1, 3), dtype=np.float32)
    faces = np.zeros((9, 3), dtype=np.uint32)
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], device="cuda")

    bvh = cubvh.cuBVH(vertices, faces)
    check_no_surface_result(bvh.unsigned_distance(points, return_uvw=True))
    check_no_surface_result(bvh.signed_distance(points, return_uvw=True, mode="watertight"))
    check_no_surface_result(bvh.signed_distance(points, return_uvw=True, mode="raystab"))


if __name__ == "__main__":
    main()
