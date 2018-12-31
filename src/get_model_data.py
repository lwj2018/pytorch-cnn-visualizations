import argparse
import time

import sys
sys.path.append("..")
sys.path.append("../TRN")
import numpy as np
import torch.nn.parallel
import torch.optim
import pdb
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
from TRN.dataset import TSNDataSet, VideoDataset
from TRN.models import TSN
from TRN.datasets_video import return_dataset
from TRN.transforms import *
from TRN.ops import ConsensusModule

def get_model_data():
    # CONFIG
    parser = argparse.ArgumentParser(
        description="TRN testing on the full validation set")
    parser.add_argument('--dataset', type=str, default="pbd-v0.2")
    parser.add_argument('--modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'], default="RGB")
    parser.add_argument('--weights', type=str, 
                    default="/mnt/data/old/liweijie/c3d_models/trn/"
                    +"nocrop_TRN_pbd-v0.2_RGB_BNInception_TRNmultiscale_segment3_checkpoint.pth.tar")
    parser.add_argument('--arch', type=str, default="BNInception")
    parser.add_argument('--save_scores', type=str, default=None)
    parser.add_argument('--test_segments', type=int, default=3)
    parser.add_argument('--max_num', type=int, default=-1)
    parser.add_argument('--test_crops', type=int, default=1)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_fusion_type', type=str, default='TRNmultiscale',
                        choices=['avg', 'TRN','TRNmultiscale'])
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--img_feature_dim',type=int, default=256)
    parser.add_argument('--num_set_segments',type=int, default=1,help='TODO: select multiply set of n-frames from a video')
    parser.add_argument('--softmax', type=int, default=0)
    args = parser.parse_args()

    categories, args.train_list, args.val_list, args.root_path, prefix = return_dataset(args.dataset, args.modality)
    num_class = len(categories)

    net = TSN(num_class, args.test_segments if args.crop_fusion_type in ['TRN','TRNmultiscale'] else 1, args.modality,
            base_model=args.arch,
            consensus_type=args.crop_fusion_type,
            img_feature_dim=args.img_feature_dim,
            )

    # checkpoint = torch.load(args.weights, map_location={'cuda:0':'cpu'})
    checkpoint = torch.load(args.weights)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)

    cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(net.input_size),
        ])

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(net.input_mean, net.input_std)
    else:
        normalize = IdentityTransform()

    data_loader = torch.utils.data.DataLoader(
            TSNDataSet(args.root_path, args.train_list, num_segments=args.test_segments,
                    new_length=data_length,
                    modality=args.modality,
                    image_tmpl=prefix,
                    random_shift=False,
                    transform=torchvision.transforms.Compose([
                        cropping,
                        Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
                        ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
                        normalize,
                    ])),
            batch_size=1, shuffle=False,
            num_workers=args.workers * 2, pin_memory=True)

    net = torch.nn.DataParallel(net.cuda())

    return net, data_loader