import math
import os
import json
import numpy as np
from PIL import Image
import torch
from skimage.measure import marching_cubes
import trimesh


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')

def get_projection_matrix(znear: float, zfar: float, fovx: float, fovy: float):
    tan_half_fovy = math.tan(fovy / 2)
    tan_half_fovx = math.tan(fovx / 2)
    
    top = tan_half_fovy * znear
    bottom = -top
    right = tan_half_fovx * znear
    left = -right
    
    P = torch.zeros(4, 4)

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -(2.0 * zfar * znear) / (zfar - znear)
    P[3, 2] = -1.0
    return P

def fov2focal(fov: float, pixels: int):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal: float, pixels: int):
    return 2 * math.atan(pixels / (2 * focal))

def create_visual_hull(
    masks: torch.Tensor, 
    world2camera_matrices: torch.Tensor, 
    camera2clip_matrices: torch.Tensor, 
    voxel_size = 0.1, 
    space_extent = 30.0, 
    debug = False
):
    assert space_extent > 0.0 and voxel_size > 0.0
    steps = int(2.0 * space_extent / voxel_size)
    x = torch.linspace(-space_extent, space_extent, steps=steps, dtype=torch.float32, device=device)
    y = torch.linspace(-space_extent, space_extent, steps=steps, dtype=torch.float32, device=device)
    z = torch.linspace(-space_extent, space_extent, steps=steps, dtype=torch.float32, device=device)
    xv, yv, zv = torch.meshgrid(x, y, z, indexing='ij')
    voxels = torch.vstack((xv.flatten(), yv.flatten(), zv.flatten(), torch.ones_like(xv.flatten()))).T
    occupancy = torch.ones(voxels.shape[0], dtype=torch.bool, device=device)

    for i, (mask, world2cam, cam2clip) in enumerate(zip(masks, world2camera_matrices, camera2clip_matrices)):
        print(f"Processing {i + 1} th data.")
        camera_coords = (world2cam @ voxels.T).T
        clip_coords = (cam2clip @ camera_coords.T).T
        ndc_coords = clip_coords[:, :3] / clip_coords[:, 3:4]
        in_frustum = (
            (torch.abs(ndc_coords[:, 0]) <= 1) & 
            (torch.abs(ndc_coords[:, 1]) <= 1) & 
            (ndc_coords[:, 2] >= 0) & 
            (ndc_coords[:, 2] <= 1)
        )
        H = mask.shape[0]
        W = mask.shape[1]
        u = ((ndc_coords[:, 0] + 1) / 2) * W
        v = ((-ndc_coords[:, 1] + 1) / 2) * H
        u = u.to(torch.int32)
        v = v.to(torch.int32)
        valid_pixel = (u >= 0) & (u < W) & (v >= 0) & (v < H) & in_frustum
        if debug:
            img = torch.zeros((H, W), device=device)
            img[v[valid_pixel], u[valid_pixel]] = 255
            img = img.detach().to(device='cpu').numpy().reshape((H, W)).astype(np.uint8)
            img = Image.fromarray(img, mode='L')
            img.save(f"render_for_debug_{i}.png")
        silhouette_check = torch.zeros(voxels.shape[0], dtype=torch.bool, device=device)
        silhouette_check[valid_pixel] = mask[v[valid_pixel], u[valid_pixel]] > 0
        occupancy &= silhouette_check
        print("sum: ", torch.sum(occupancy))

    volume = occupancy.reshape(xv.shape).detach().to('cpu').numpy()
    verts, faces, _, _ = marching_cubes(volume=volume, level=0.5)
    verts = verts * voxel_size - space_extent
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    return mesh

data_dir = "sample_data"
camera_params_path = os.path.join(data_dir, "transforms_train.json")

with open(camera_params_path, 'r') as f:
    camera_json = json.load(f)
frames = camera_json["frames"]
fov = camera_json["camera_angle_x"]
mask_image_list = []
world2camera_list = []
camera2clip_list = []
for frame in frames:
    image_file_path = frame["file_path"]
    image = Image.open(os.path.join(data_dir, image_file_path))
    w, h = image.size
    mask_image = np.array(Image.open(os.path.join(
        data_dir, 'image_mask', image_file_path.split('/')[-1])))
    mask_image_list.append(mask_image)
    fovx = fov
    fovy = focal2fov(fov2focal(fovx, w), h)
    camera2clip = \
        get_projection_matrix(
            znear=0.01, 
            zfar=100.0, 
            fovx=fovx, 
            fovy=fovy
        )
    camera2clip = np.array(camera2clip, dtype=np.float32).reshape(4, 4)
    camera2clip_list.append(camera2clip)
    c2w = np.array(frame["transform_matrix"], dtype=np.float32)
    world2camera_list.append(np.linalg.inv(c2w))

mesh = create_visual_hull(
    torch.tensor(np.array(mask_image_list), device=device), 
    torch.tensor(np.array(world2camera_list), device=device), 
    torch.tensor(np.array(camera2clip_list), device=device)
)
# mesh.show()
mesh.export(os.path.join(data_dir, "visual_hull_mesh.obj"))
