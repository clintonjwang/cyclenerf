import numpy as np
import torch
th = torch
nn = torch.nn
F = nn.functional


"""
Methods relating to cameras and camera views (poses).
A camera view is specified by:
- K: [3,4]. camera intrinsic matrix
- c2w: [4,4]. camera to world matrix (pose)
"""

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def get_camera_intrinsics(f, fov):
    cx, cy = 0,0
    K = np.array([[fx, 0., cx, 0.],
               [0., fy, cy, 0],
               [0., 0, 1, 0],
               [0, 0, 0, 1]])
    return K

def pose_spherical(theta, phi=-30., radius=4.):
    """
    Args:
        theta: float. horizontal angle in degrees
        phi: float. elevation angle in degrees
        radius: float. distance from origin
    Returns:
        c2w: [4,4]. camera pose
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def get_coords_at_ray_depths(rays, depth):
    return

def get_rays_for_view(view, args):
    """
    Args:
        view: [K,c2w]. camera parameters
    Returns:
        c2w: [4,4]. camera pose
    """
    h,w = args.img_dims
    fov = view[7]
    focal = view[8]
    dirs = generate_ray_directions(w,h, fov, focal)
    cam_pos = view[:3]
    cam_dir = view[3:6]
    img_center = cam_pos + cam_dir * focal
    start_coords = torch.linspace()

def get_rays(H, W, K, c2w):
    """
    Args:
        H,W: ints. height, width
        K: [3,4]. camera intrinsic matrix
        c2w: [4,4]. camera to world matrix
    Returns:
        rays_o: [?]. ray origins
        rays_d: [?]. ray directions
    """
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

# def get_rays_np(H, W, K, c2w):
#     i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
#     dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
#     # Rotate ray directions from camera frame to the world frame
#     rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # Translate camera frame's origin to the world frame. It is the origin of all rays.
#     rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
#     return rays_o, rays_d
