# -*- coding:utf-8 -*-

# from __future__ import division
# from __future__ import absolute_import
# from __future__ import print_function


import torch
from .bbox_utils import decode, nms
from torch.autograd import Function

from data.config import cfg


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    num_classes = cfg.NUM_CLASSES
    top_k = cfg.KEEP_TOP_K
    nms_top_k = cfg.NMS_TOP_K
    nms_thresh = cfg.NMS_THRESH
    conf_thresh = cfg.CONF_THRESH
    variance = cfg.VARIANCE

    @staticmethod
    def forward(loc_data, conf_data, prior_data):
        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        output = torch.zeros(num, Detect.num_classes, Detect.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, Detect.num_classes).transpose(2, 1)

        batch_priors = prior_data.view(-1, num_priors, 4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        decoded_boxes = decode(loc_data.view(-1, 4), batch_priors, Detect.variance)
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()

            for cl in range(1, Detect.num_classes):
                c_mask = conf_scores[cl].gt(Detect.conf_thresh)
                scores = conf_scores[cl][c_mask]

                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes_, scores, Detect.nms_thresh, Detect.nms_top_k)
                count = count if count < Detect.top_k else Detect.top_k
                output[i, cl, :count] = torch.cat(
                    (scores[ids[:count]].unsqueeze(1), boxes_[ids[:count]]), 1
                )
        return output
