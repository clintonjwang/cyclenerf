import torch
th = torch
nn = torch.nn
F = nn.functional

def estimate_field_from_img(img, view, DP):
    """
    Args:
        img: [1, 3, h, w]. RGB tensor image
        view: [7]. Camera position, camera direction, focal length.
        DP: nn.Module. Depth predictor network
    Returns:
        RF: [num_rays, 3]. (x,y,z, phi,theta) -> (r,g,b,o)
    """
    h,w = img.shape[2:]
    depths = DP(img)
    with torch.no_grad():
        rays = camera.get_rays_for_view(view, args=args)
        RF = RadianceField(rays, view, depths)
    return RF

def estimate_bg_color(img):
    np_img = img.detach().cpu().squeeze().numpy()
    corners = [np_img[:, :h//4, :w//4], np_img[:, 3*h//4:, :w//4],
                np_img[:, :h//4, 3*w//4:], np_img[:, 3*h//4:, 3*w//4:]]
    corners = np.concatenate([a.reshape(3,-1) for a in corners], axis=1)
    bg_color = np.median(corners, axis=1)
    return bg_color

class RadianceField:
    """
    Radiance fields are lists of plenoxels, along with a map of which voxels have not observed.
    Ignore view-dependent effects for now. Each plenoxel is a 7-dim vector (world_pos, rgba)
    """
    def __init__(self, rays, view, depths):
        r"""
        Args:
            rays: [HW, 6]. pos, dir of the rays for each pixel.
            view: [7]. camera intrinsics + pose.
            depths: [HW, 1]. depths at each pixel.
        """
        self.bg_color = estimate_bg_color(img)
        self.orig_view = view
        self.orig_view_rays = rays
        self.orig_view_depths = depths

        coords = camera.get_coords_at_ray_depths(rays, depths, view)
        flat_img = img.reshape(1,3,h*w)
        xyzrgb = torch.cat([coords, flat_img], axis=1) # [HW, 6]. xyz and rgb values per pixel.

        # self.empty_space = BinaryMap()
        # self.minifield_classifier = ClassMap()
        self.plenoxels = torch.cat([rgbxyz, torch.ones_like(rgbxyz)], axis=1)

    def render_new_view(self, view, transformer):
        rays = camera.get_rays_for_view(view)
        terminal_rgb, terminal_indices, unseen_coords, unseen_rgba = self._get_terminal_and_unseen_coords_for_rays(rays)
        new_img = torch.empty(device=view.device, dtype=torch.float16)
        transformer.estimate_coords(unseen_coords, self.plenoxels)

    def _get_terminal_and_unseen_coords_for_rays(self, rays):
        r"""
        Args:
            rays: [B, 6]. world coordinates and direction vectors of rays.
        Return:
            terminal_rgb: [Bt, 3]. RGB (0-1) of rays that terminate
            terminal_indices: [Bt] int. ray indices of the rays that terminate
            unseen_rays: [B-Bt, 6]. world coordinates and direction vectors of rays that encounter unseen space
            unseen_rgba: [B-Bt, 4]. accumulated rgba of unseen_rays
        """
        coord = rays[:3] # [B, 3]. starting coords
        orig_uv_duv_coord = camera.world2img_coords(coord, self.orig_view) # [B, 4]. u,v coordinates of these coordinates
        corresponding_rays = self.orig_view_rays # [B]. index of closest ray in the original view
        ray_d = rays[3:]
        coord += ray_d

        terminal_rgb
        terminal_indices
        unseen_rays
        unseen_rgba
        return terminal_rgb, terminal_indices, unseen_rays, unseen_rgba

    def _step_coords_along_rays(coords, orig_view_rays, ray_d):
        r"""
        Args:
            coords: [B, 6]. world coordinates along rays.
            orig_view_rays: [B, 6]. rays (pos, dir) of the original view that are closest to the coords.
            ray_d: [B, 3]. direction vectors of the rays.
        Return:
            next_coords: [B, 6].
            orig_view_next_rays: [B, 6]. the next rays (pos, dir) of the original view that are closest to the coords.
        """
        
        orig_view_next_rays


    # def cluster_depth():
    #     # for minifield approach
    #     return

    # def _get_unknown_coord_along_ray(self, ray, view):
    #     
    #     coord = self._step_coord_along_ray(coord, ray)
    #     accumulate_rendered_rays()
    #     return 


class BinaryMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3,512), nn.ReLU(),
            nn.Linear(512,128), nn.ReLU(), nn.Linear(128,1))
    def forward(self, xyz):
        return self.mlp(xyz)

class FieldEmbedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, RF):
        return
