This repo contains code to reproduce vessel segmentation masks from the manuscript **"Bessel Beam Optical Coherence Microscopy Enables Multiscale Assessment of Cerebrovascular Network Morphology and Function."**

# Installation

Create a new virtual environment using, for example, anaconda:

    conda create -n octaunet python=3.8

Install relevant packages:

    pip install torch==1.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

    pip install monai==1.0.1

    pip install SimpleITK scikit-image scikit-learn tqdm

# Data & Models

Make sure to have Git LFS installed:

    git lfs install

Download the data (labeled test, val, and train volume) and the trained models (checkpoints) from:

    git clone https://huggingface.co/bwittmann/OCTA-unet

Move the trained models to the `./runs` folder. The structure should follow:

    └── runs/
        └── manual_annotated/               # solely trained on our manually annotated volume
        └── manual_annotated_synthetic/     # pre-trained on synthetic data & finetuned on our manually annotated volume
        └── synthetic/                      # solely trained on synthetic data

Move the splits to the `./dataset` folder. The structure should follow:

    └── dataset/
        └── splits/
            └── test_data.nii               # test volume
            └── test_label.nii
            └── val_data.nii                # val volume
            └── val_label.nii
            └── train_data.nii              # train volume
            └── train_label.nii

# Run

To test the performance of our provided models (`manual_annotated_synthetic`, `manual_annotated`, `synthetic`) on our provided volumes, please run:

    python inference.py --ckpt manual_annotated_synthetic --data_folder <path_to_splits> --test

To performance inference on unlabeled data, please run:

    python inference.py --ckpt manual_annotated_synthetic --data_folder <path_to_data>
