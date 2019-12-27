# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import site
import torch


class Config(object):
    # GPU config
    gpu_num = "0"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + gpu_num if use_cuda else "cpu")
    if use_cuda:
        torch_def_tensor = "torch.cuda.FloatTensor"
    else:
        torch_def_tensor = "torch.FloatTensor"

    # ratio for resizing frames small
    res_ratio = 2

    # OpenCV detection config
    oc_color = (255, 0, 0)
    # xml file path for OpenCV haarcascade Classifier
    cascade_path = os.path.join(
        site.getsitepackages()[0], "cv2/data/haarcascade_frontalface_alt.xml"
    )

    # dumped model path for face detection
    fb_path = "weights/Faceboxes.pth"
    fb_thresh = 0.5

    mask_img_path = "img/santa_hat.png"


def get_config():
    config = Config()
    return config
