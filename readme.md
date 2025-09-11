# cuBVH

A CUDA Mesh BVH acceleration toolkit.

### Install

Make sure `torch` and CUDA are installed first.

```bash
pip install git+https://github.com/ashawkey/cubvh --no-build-isolation

# or locally
git clone --recursive https://github.com/ashawkey/cubvh
cd cubvh
pip install . --no-build-isolation
```
It will take several minutes to build the CUDA dependency.

#### Trouble Shooting
**`fatal error: eigen/matrix.h: No such file or directory`**

This is a known issue for `torch==2.1.0` and `torch==2.1.1` (https://github.com/pytorch/pytorch/issues/112841). 
To patch up these two versions, clone this repository, and copy `patch/eigen` to your pytorch include directory:
```bash
# for example, if you are using anaconda (assume base env)
cp -r patch/eigen ~/anaconda3/lib/python3.9/site-packages/torch/include/pybind11/
```

**`fatal error: Eigen/Dense: No such file or directory`**

Please make sure [`eigen >= 3.3`](https://eigen.tuxfamily.org/index.php?title=Main_Page) is installed. 
We have included it as a submodule in this repository, but you can also install it in your system include path.
(For example, ubuntu systems can use `sudo apt install libeigen3-dev`.)

### Usage

**Basics:**

```python
import numpy as np
import trimesh
import torch

import cubvh

### build BVH from mesh
mesh = trimesh.load('example.ply')
# NOTE: you need to normalize the mesh first, since the max distance is hard-coded to 100.
BVH = cubvh.cuBVH(mesh.vertices, mesh.faces) # build with numpy.ndarray/torch.Tensor

### query ray-mesh intersection
rays_o, rays_d = get_ray(pose, intrinsics, H, W) # [N, 3], [N, 3], query with torch.Tensor (cuda)
intersections, face_id, depth = BVH.ray_trace(rays_o, rays_d) # [N, 3], [N,], [N,]

### query unsigned distance
points # [N, 3]
# uvw is the barycentric corrdinates of the closest point on the closest face (None if `return_uvw` is False).
distances, face_id, uvw = BVH.unsigned_distance(points, return_uvw=True) # [N], [N], [N, 3]

### query signed distance (INNER is NEGATIVE!)
# for watertight meshes (default)
distances, face_id, uvw = BVH.signed_distance(points, return_uvw=True, mode='watertight') # [N], [N], [N, 3]
# for non-watertight meshes:
distances, face_id, uvw = BVH.signed_distance(points, return_uvw=True, mode='raystab') # [N], [N], [N, 3]
```

**Robust Mesh Occupancy:**

UDF + flood-fill for possibly non-watertight/single-layer meshes:

```python
import torch
import cubvh
import numpy as np

resolution = 512
device = torch.device('cuda')

BVH = cubvh.cuBVH(vertices, faces)

grid_points = torch.stack(
    torch.meshgrid(
        torch.linspace(-1, 1, resolution, device=device),
        torch.linspace(-1, 1, resolution, device=device),
        torch.linspace(-1, 1, resolution, device=device),
        indexing="ij",
    ), dim=-1,
) # [N, N, N, 3]

# query dense UDF
udf, _, _ = BVH.unsigned_distance(grid_points.view(-1, 3), return_uvw=False)
udf = udf.view(opt.res, opt.res, opt.res).contiguous()

# floodfill to get SDF
occ = udf < 2 / resolution # tolerance 2 voxel
floodfill_mask = cubvh.floodfill(occ)
empty_label = floodfill_mask[0, 0, 0].item()
empty_mask = (floodfill_mask == empty_label)
occ_mask = ~empty_mask
sdf = udf - eps  # inner is negative
inner_mask = occ_mask & (sdf > 0)
sdf[inner_mask] *= -1

sdf = sdf.cpu().numpy()

```
Check [`test/extract_mesh_watertight.py`](test/extract_mesh_watertight.py) for more details.


### Acknowledgement

* Credits to [Thomas Müller](https://tom94.net/)'s amazing [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and [instant-ngp](https://github.com/NVlabs/instant-ngp)!
