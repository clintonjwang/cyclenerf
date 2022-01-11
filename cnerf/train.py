import os, argparse, yaml, pdb
osp = os.path

import torch
th = torch
nn = torch.nn
F = nn.functional
import bitsandbytes as bnb

from cnerf import args as cargs
from cnerf import camera, losses
import cnerf.datasets
import cnerf.depth
import cnerf.gan
from cnerf.transformer import RFtransformer

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.config = yaml.load(open(args.config_path, "r"))
    loss_weights = args.loss_weights

    pdb.set_trace()
    dataloader = cnerf.datasets.get_dataloader(args)

    camera_estimator = camera.get_camera_estimator()
    DP = cnerf.depth.load_depth_predictor("MiDaS_small")
    discriminator = cnerf.gan.Discriminator()
    transformer = RFtransformer()

    generator_params = list(camera_estimator.parameters()) + list(DP.parameters()) + list(transformer.parameters())
    G_optim = bnb.optim.Adam8bit(generator_params, lr=args.lr["generator"])
    D_optim = bnb.optim.Adam8bit(discriminator.parameters(), lr=args.lr["discrim"])

    C = 3
    H,W = 10,10
    iteration = 1

    loss_history = {}
    for epoch in range(1,args.epochs):
        # for real_img in dataloader: # (1,C,H,W)
        for real_img in [torch.randn(1,C,H,W).cuda()]:
            orig_view = torch.tensor((0.,0.,-5., #camera at (0,0,-5)
                    0.,0.,1., # facing (0,0,1.)
                    fov, 1.), device=real_img.device) # with focal length 1
            fov, f, theta, phi = camera_estimator(img)
            K = (fov, f)
            c2w = pose_spherical(theta, phi, radius)

            with torch.cuda.amp.autocast(): # Casts operations to mixed precision
                RF = radiance.estimate_field_from_img(real_img, orig_view, DP=DP)
                new_view = camera.generate_new_view(iteration=iteration)
                img_new_view = render.render_field_from_view(RF, new_view, transformer=transformer)

                D_fake_logit = discriminator(img_new_view)
                D_real_logit = discriminator(real_img)

                D_loss = D_fake_logit - D_real_logit
                D_optim.zero_grad()
                D_loss.backward()
                D_optim.step()
                losses.record_loss("D loss", D_loss.item(), loss_history)

                if args.recon_rf_phase[0] <= epoch <= args.recon_rf_phase[1]:
                    recon_RF = transformer.autoencode_field(RF)
                    recon_rf_loss = RF.autoencode_loss(recon_RF)

                    loss += loss_weights["recon RF"] * recon_rf_loss
                    losses.record_loss("recon RF", recon_rf_loss.item(), loss_history)

                if args.recon_img_phase[0] <= epoch <= args.recon_img_phase[1]:
                    recon_img = render.render_field_from_view(recon_RF, orig_view, transformer=transformer)
                    recon_img_l1_loss = losses.l1_loss(img, recon_img)
                    recon_img_percept_loss = losses.perceptual_loss(img, recon_img)
                    losses.perceptual_loss(img, recon_img)

                    loss += loss_weights["L1 recon img"] * recon_img_l1_loss
                    losses.record_loss("L1 recon img", recon_img_l1_loss.item(), loss_history)
                    loss += loss_weights["perceptual recon img"] * recon_img_percept_loss
                    losses.record_loss("perceptual recon img", recon_img_percept_loss.item(), loss_history)

                if args.discrim_phase[0] <= epoch <= args.discrim_phase[1]:
                    G_loss = -D_fake_logit * loss_weights["discrim"]
                    losses.record_loss("discrim loss", -D_fake_logit.item(), loss_history)

                if args.cycle_phase[0] <= epoch <= args.cycle_phase[1]:
                    with torch.no_grad():
                        new_RF = radiance.estimate_field_from_img(img_new_view, new_view, DP=DP)
                    cycle_img = render.render_field_from_view(new_RF, orig_view, transformer=transformer)
                    cycle_img_l1_loss = losses.l1_loss(recon_img, cycle_img)
                    cycle_img_percept_loss = losses.perceptual_loss(recon_img, cycle_img)
                    
                    loss += loss_weights["L1 cycle img"] * cycle_img_l1_loss
                    losses.record_loss("L1 cycle img", cycle_img_l1_loss.item(), loss_history)
                    loss += loss_weights["perceptual cycle img"] * cycle_img_percept_loss
                    losses.record_loss("perceptual cycle img", cycle_img_percept_loss.item(), loss_history)

                G_optim.zero_grad()
                loss.backward()
                G_optim.step()

            if iteration >= args.iterations:
                break
            iteration += 1

if __name__ == "__main__":
    args = cargs.parse_args()
    main(args)
