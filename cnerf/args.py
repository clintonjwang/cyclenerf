import sys
import os, shutil
osp = os.path

import argparse

def parse_args(callback=None, training=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", type=str, default="manual")
    parser.add_argument("--config_dir", type=str, default=osp.expanduser(f"~/code/cyclenerf/configs"))
    parser.add_argument(
        "--exp_group_name", type=str, default=None,
        help="if we want to group some experiments together",
    )
    parser.add_argument(
        "--results_dir", type=str, default=osp.expanduser(f"~/code/cyclenerf/results"),
    )
    parser.add_argument("--ray_batch_size", "-R", type=int, default=50000, help="ray batch size")
    parser.add_argument("--color_space", type=str, default="RGB", choices=["RGB", "YUV", "HSL"])
    parser.add_argument("--seed", type=int, default=0)

    if training:
        parser.add_argument(
            "--checkpoints_dir", type=str, default=osp.expanduser(f"~/code/cyclenerf/checkpoints"),
        )
        parser.add_argument("--pretrained_path", type=str, default=None)
        parser.add_argument(
            "--dataset", type=str, default="clevr", choices=["clevr", "coco"]
        )
        parser.add_argument(
            "--data_dir", type=str, default="/data/vision/polina/scratch/clintonw/datasets",
        )
        # parser.add_argument("--batch_size", type=int, default=1, help="image batch size")
        parser.add_argument("--mlp_type", type=str, default="mlp", choices=["mlp", "resnet"])
        parser.add_argument("--iterations", type=int, default=1000)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--train_frac", type=float, default=.9)
        parser.add_argument("--lr_depth", type=float, default=1e-7)
        parser.add_argument("--lr_discrim", type=float, default=1e-6)
        parser.add_argument("--lr_transformer", type=float, default=1e-5)

    else:
        parser.add_argument("--camera_path", type=str)
        parser.add_argument(
            "--img_path", type=str, default="/data/vision/polina/scratch/clintonw/datasets",
        )
        parser.add_argument(
            "--batch_size", type=int, default=1, help="view batch size"
        )

    if callback is not None:
        parser = callback(parser)
    args = parser.parse_args()

    if not osp.exists(args.data_dir):
        raise ValueError(f"bad data directory {args.data_dir}")


    if args.exp_group_name is not None:
        args.results_dir = osp.join(args.results_dir, args.exp_group_name)
        args.checkpoints_dir = osp.join(args.checkpoints_dir, args.exp_group_name)

    args.results_dir = osp.join(args.results_dir, args.name)
    args.checkpoints_dir = osp.join(args.checkpoints_dir, args.name)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    args.config_path = osp.join(args.config_dir, args.name+".yaml")
    if not osp.exists(args.config_path):
        args.config_path = osp.join(args.config_dir, "default.yaml")
    shutil.copy(args.config_path, osp.join(args.results_dir, "config.yaml"))

    args.recon_phase = (0,200)
    args.cycle_phase = (200,-1)
    args.discrim_phase = (100,-1) # train the to fool the discriminator

    args.lr = {
        "depth": args.lr_depth,
        "discrim": args.lr_discrim,
        "transformer": args.lr_transformer,
    }

    print("EXPERIMENT NAME:", args.name)
    print("* Config file:", args.config_path)
    print("* Dataset:", args.data_dir)
    print("* Dataset:", args.data_dir)
    return args
