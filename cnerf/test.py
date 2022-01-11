import os, argparse, yaml, pdb
osp = os.path

import torch
th = torch
nn = torch.nn
F = nn.functional

from cnerf import args as cargs
from cnerf import datasets
from cnerf import rays
from cnerf.models import DepthPredictor, NeRFHyperNet

def main(args):
    args.config = yaml.load(open(args.config_path, "r"))

    pdb.set_trace()
    dataloader = datasets.get_dataloader(args)

    optimizer = nn.optim.Adam(model.parameters())
    B = 1 #args.batch_size
    C = 3
    H,W = 10,10
    imgs = torch.randn(B,C,H,W) # (B,C,H,W)


    depth = DepthPredictor(imgs) # (B,H,W)
    orig_view = torch.tensor((0.,0.,-5., #camera at (0,0,-5)
            0.,0.,1., # facing (0,0,1.)
            1., )) # with focal length 1

    NeRF = NeRFHyperNet(imgs) # (x,y,z, phi,theta) -> (r,g,b,o)
    rays = rays.construct_rays_from_camera(orig_view) # 
    pred_img = NeRF.estimate_img_from_rays(rays)

    
    new_view = torch.tensor((0.,0.,-4.,1.))
    rays = rays.construct_rays_from_camera(new_view) # 
    pred_img = NeRF.estimate_img_from_rays(rays)

    # for imgs in dataloader: # (B,C,H,W)
    #   return


if __name__ == "__main__":
    args = cargs.parse_args()
    main(args)
