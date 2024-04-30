"""File containing model related functionality."""

import torch.nn as nn
import numpy as np
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from sklearn.metrics import confusion_matrix, roc_auc_score
from skimage.morphology import skeletonize, skeletonize_3d


def get_model(config):
    model = OctSegNet(config)
    return model

def get_loss_dict():
    loss = {
        'seg': DiceLoss()
    }
    return  loss

def cl_dice(v_p, v_l):
    def cl_score(v, s):
        return np.sum(v*s)/np.sum(s)

    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

def estimate_metrics(pred, gt, threshold=0.5):
    pred_labels = (pred >= threshold).float()
    metrics = {}

    tn, fp, fn, tp = confusion_matrix(
        gt.flatten().cpu().numpy(), 
        pred_labels.flatten().cpu().numpy(), 
        labels=[0, 1]
    ).ravel()

    auc = roc_auc_score(
        gt.cpu().numpy().flatten(),
        pred.cpu().numpy().flatten()
    )

    cldice = cl_dice(
        pred_labels.squeeze().cpu().clone().detach().byte().numpy(), 
        gt.squeeze().cpu().clone().detach().byte().numpy()
    )

    metrics['tpr'] = tp / (tp + fn) # recall
    metrics['fpr'] = fp / (fp + tn)
    metrics['precision'] = tp / (tp + fp)
    metrics['dice'] = (2 * tp) / (2 * tp + fp + fn)
    metrics['accuracy'] = (tp + tn) / (tn + fp + tp + fn)
    metrics['auc'] = auc
    metrics['cldice'] = cldice

    return metrics


class OctSegNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config

        # init unet 
        if config['model'] == 'unet':
            self._unet = UNet(
                spatial_dims=3, in_channels=1, out_channels=1,
                channels=config['unet']['channels'], strides=config['unet']['strides'],
                act=config['unet']['activation'], norm=config['unet']['normalization'], 
                dropout=config['dropout'], 
            )
        else:
            raise NotImplementedError   

        # get loss functions
        self._loss_functions = get_loss_dict()

    def forward(self, x, y=None, epoch=0, inference=False):
        # predict
        x_out = self._unet(x).sigmoid()

        if inference:
            return x_out

        # estimate losses
        loss_dict = {}
        loss_dict['seg'] = self._loss_functions['seg'](x_out, y)
        return loss_dict, x_out
