# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import cv2
import time
import numpy as np
import argparse
import torch
import torch.backends.cudnn as cudnn

from config import get_config

from faceboxes import FaceBox
from data.config import cfg
from utils.augmentations import FaceBoxesBasicTransform


def main(args, config):
    print("# Startng recording with a camera")
    # cap = cv2.VideoCapture(config.camera_id)
    cap = cv2.VideoCapture(0)

    print("# Load santa hat PNG img")
    mask_img = cv2.imread(config.mask_img_path, -1)

    if args.face_detector == "fb":
        torch.set_default_tensor_type(config.torch_def_tensor)

        fb = get_faceboxes(config.fb_path)
        fb.to(config.device)
        if config.use_cuda:
            cudnn.benchmark = True
        print("# Faceboxes face detection model is ready")

    elif args.face_detector == "opencv":
        # get Face Detector with OpenCV
        cascade = cv2.CascadeClassifier(config.cascade_path)
        print("# OpenCV CascadeClassifier face detection  model is ready")

    print("# Start App.\n")
    while True:
        # get a capture image from the camera
        ret, frame = cap.read()

        # make the frame resize smaller in order to process the face recognition faster.
        orgHeight, orgWidth = frame.shape[:2]
        frame_size = (
            int(orgWidth / config.res_ratio),
            int(orgHeight / config.res_ratio),
        )
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)

        # Face detect with OpenCV's cascade_detector
        if args.face_detector == "opencv":
            try:
                left_up, right_bottom = detect_with_cascade(
                    frame, cascade, mask_img, args
                )
            except Exception as e:
                print("#### No Face detected by detect_with_cascade", e.args)

        # Face detect with Faceboxes
        elif args.face_detector == "fb":
            try:
                left_up, right_bottom = detect_with_faceboxes(
                    fb, frame, config.fb_thresh, config.device, mask_img, args
                )
            except Exception as e:
                print("#### No Face detected by faceboxes detector", e.args)

        # finish detection when you type "q" key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # finish the detection
    cv2.destroyWindow("frame")
    cap.release()
    cv2.destroyAllWindows()


def detect_with_cascade(frame, cascade, mask_img, args):
    t1 = time.time()
    # get face points. return [[x, y, wigth, length], ...]
    facerect = cascade.detectMultiScale(
        frame, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10)
    )

    # put rectagles along with the detected faces
    if len(facerect) > 0:
        for rect in facerect:
            x, y, w, h = rect
            left_up, right_bottom = (x, y), (x + w, y + h)
            if args.rect:
                cv2.rectangle(
                    frame, left_up, right_bottom, config.oc_color, thickness=2
                )
            if args.santa:
                frame = combine_img(frame, mask_img, left_up, right_bottom)

    t2 = time.time()
    print("#### Elapsed time for detecting one frame:{}".format(t2 - t1))
    # show frame
    cv2.imshow("frame", frame)
    return left_up, right_bottom


def detect_with_faceboxes(net, frame, thresh, device, mask_img, args):
    frame = np.array(frame, copy=True)
    img_to_net = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
    height, width, _ = img_to_net.shape

    x = FaceBoxesBasicTransform(img_to_net)

    x = torch.from_numpy(x).unsqueeze(0).to(device)

    t1 = time.time()
    with torch.no_grad():
        y = net(x)
    detections = y.data

    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):
        for j in range(detections.size(2)):
            if detections[0, i, j, 0] >= thresh:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(int)
                left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
                if args.rect:
                    cv2.rectangle(frame, left_up, right_bottom, (0, 0, 255), 2)
                if args.santa:
                    frame = combine_img(frame, mask_img, left_up, right_bottom)

    t2 = time.time()
    print("#### Elapsed time for detecting one frame:{}".format(t2 - t1))

    cv2.imshow("frame", frame)
    return left_up, right_bottom


def get_faceboxes(model_path):
    net = FaceBox(cfg, "test")
    net.load_state_dict(torch.load(model_path, lambda storage, loc: storage))
    net.eval()
    return net


def scale_to_width(img, width):
    scale = width / img.shape[1]
    return cv2.resize(img, dsize=None, fx=scale, fy=scale)


# def combine_img(frame, img, x, y):
def combine_img(frame, mask_img, left_up, right_bottom):
    # https://cif-lab.hatenadiary.jp/entry/2018/05/06/214829
    # height_over_check = lambda x: self.HEIGHT if x > self.HEIGHT else x
    # width_over_check = lambda x: self.WIDTH if x > self.WIDTH else x

    # img = cv2.resize(img, (int(self.WIDTH / 1.2), int(self.HEIGHT / 1.2)))
    img_width = right_bottom[0] - left_up[0]
    mask_img = scale_to_width(mask_img, img_width)
    mheight, mwidth = mask_img.shape[:2]

    mheight = left_up[1] if (left_up[1] - mheight) < 0 else mheight

    # ex = width_over_check(x + width)
    # ey = height_over_check(y + height)

    mask = mask_img[:, :, 3]  # これでアルファチャンネルのみの行列が抽出。
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask = mask / 255.0

    mask_img = mask_img[:, :, :3]
    frame_float = frame.astype(np.float64)

    frame_float[left_up[1] - mheight : left_up[1], left_up[0] : right_bottom[0]] *= (
        1.0 - mask[-mheight - 1 : -1, :]
    )
    frame_float[left_up[1] - mheight : left_up[1], left_up[0] : right_bottom[0]] += (
        mask_img[-mheight - 1 : -1, :] * mask[-mheight - 1 : -1, :]
    )

    # frame_float[y:ey, x:ex] *= 1 - mask[: (ey - y), : (ex - x)]
    # frame_float[y:ey, x:ex] += (
    #    img[: (ey - y), : (ex - x)] * mask[: (ey - y), : (ex - x)]
    # )

    frame_float = frame_float.astype(np.uint8)
    return frame_float


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--face_detector",
        "-fd",
        choices=["fb", "opencv"],
        default="fb",
        help="choice face detector",
    )
    parser.add_argument(
        "--rect", action="store_true", help="add it, surround your face",
    )
    parser.add_argument(
        "--santa", action="store_true", help="add it, put santa hat on your head"
    )

    args = parser.parse_args()
    config = get_config()
    main(args, config)
