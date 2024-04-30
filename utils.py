"""Utils functions."""

import json

import numpy as np
from skimage import io
import SimpleITK as sitk


def write_nifti(data, file_path):
    meta_data = {}
    meta_data['itk_spacing'] = [1, 1, 1]

    data_itk = sitk.GetImageFromArray(data)
    data_itk.SetSpacing(meta_data['itk_spacing'])

    sitk.WriteImage(data_itk, str(file_path))

def read_nifti(path):
    sitk_img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(sitk_img)

def read_tiff(file_path):
    img = io.imread(file_path)
    return np.array(img)

def read_json(file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
    return data
