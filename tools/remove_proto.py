# python tools/remove_proto.py --source work_dirs/vitb_prt_6dset_fp16_ft_hard_css/epoch_70.pth
import torch
import os
import argparse
import copy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str)
    parser.add_argument('--target', type=str, default=None)
    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    if args.target is None:
        args.target = '/'.join(args.source.split('/')[:-1])

    ckpt = torch.load(args.source, map_location='cpu')
    
    new_ckpt = copy.deepcopy(ckpt)
    state_dict = new_ckpt['state_dict']

    keys = ckpt['state_dict'].keys()

    proto_key = 'proto_head.kpt_prototype.prototypes'
    if proto_key in keys:
        print(f'removing {proto_key}')
        new_ckpt['state_dict'].pop(proto_key)
    
    torch.save(new_ckpt, args.source.replace('.pth', '_no_proto.pth'))

if __name__ == '__main__':
    main()
