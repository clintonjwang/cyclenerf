import os
osp = os.path
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision

def get_dataloaders(args):
    if args.color_space == "RGB":
        pass
    elif args.color_space == "YUV":
        raise NotImplementedError(args.color_space)
    else:
        raise NotImplementedError(args.color_space)

    data_dir = osp.join(osp.expanduser(args.data_dir), args.dataset)

    if "coco" in data_dir:
        raise NotImplementedError("coco")
        ds = torchvision.datasets.CocoDetection(data_dir)
        train_ds = Subset(ds, train_indices)
        train_dataloader = DataLoader(ds, batch_size=args.batch_size)
    elif "clevr" in data_dir:
        ds = ClevrDataset(data_dir)
        N = len(imgs)
        indices = list(range(N))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[:N*args.train_frac], indices[N*args.train_frac:]
        train_ds = Subset(ds, train_indices)
        train_dataloader = DataLoader(train_ds, batch_size=1)
        val_ds = Subset(ds, val_indices)
        val_dataloader = DataLoader(val_ds, batch_size=1)
    else:
        raise NotImplementedError(f"cannot handle dataset at {data_dir}")

    return train_dataloader, val_dataloader

class ClevrDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        imgs = np.load(osp.join(data_dir, "/clevr_training_data_128.npz"))["ims"]
        self.imgs = torch.tensor(imgs / 255., dtype=torch.float16)
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        return imgs[idx]


# def YUV_dataloader(DataLoader):
#     def __init__(self, args):
#         return
