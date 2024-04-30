"""Script to run inference on raw data."""

import argparse
import random
from pathlib import Path

import torch
import numpy as np
from monai.inferers import sliding_window_inference

from utils import read_json, read_tiff, write_nifti, read_nifti
from models import get_model, estimate_metrics


def inference(args):
    # load ckpt and model
    path_to_ckpt = Path('./runs/' + args.ckpt)
    device = torch.device('cuda:' + str(args.num_gpu)) if args.num_gpu >= 0 else 'cpu'
    config = read_json(path_to_ckpt / 'config.json')

    model_weights_path = path_to_ckpt / 'model.pt'
    model = get_model(config['models']).to(device=device)

    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # load data
    files = [file for file in Path(args.data_folder).glob('**/*') if file.is_file() and file.suffix in ['.tif', '.nii']]
    for file in files:
        if 'seg' in str(file.parts[-1]) or 'label' in str(file.parts[-1]):  # don't process labels and already segmented files
            continue

        try:
            data = read_tiff(file).astype(np.float32)[None, None]
        except:
            data = read_nifti(file).astype(np.float32)[None, None]
        data = torch.tensor((data - data.min()) / (data.max() - data.min()))

        # make prediction
        with torch.no_grad():
            out = sliding_window_inference(
                    inputs=data.to(device=device),
                    roi_size=config['augmentation']['patch_size'],
                    sw_batch_size=config['batch_size'],
                    predictor=model,
                    overlap=0.9,
                    mode=config['sliding_window_mode'],
                    progress=True,
                    inference=True
                )
            
        if args.test:
            try:
                split = file.stem.split('_')[0]
                label = read_nifti(file.parent / f'{split}_label.nii')

                # estimate metrics
                metrics = estimate_metrics(out.squeeze(), torch.tensor(label.squeeze()))

                print('\n', f'ckpt: {args.ckpt}', '\n', f'split: {split}')
                print(metrics)
            except:
                print('No labels provided.')

        run_name = args.ckpt.split('/')[-1]
        write_nifti((out >= args.threshold).int().squeeze().cpu().numpy(), str(file.with_suffix('')) + f'_{run_name}_{args.threshold}_seg.nii.gz')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # to get reproducable results
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # get args and config
    parser.add_argument('--data_folder', type=str, required=True, help='Path to raw data. A folder full of files.')
    parser.add_argument('--ckpt', required=True, type=str, help='Name of experiment in /runs.')
    parser.add_argument('--num_gpu', type=int, default=0, help='Id of GPU to run inference on; -1 CPU.')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    # run inference
    inference(args)