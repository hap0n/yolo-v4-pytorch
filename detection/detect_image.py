import torch
from torchvision.utils import draw_bounding_boxes
import time
from data.names import COCO_NAMES
import os
import torchvision

from utils import non_maximum_suppression
from models.yolo import YOLOv4


class YOLOv4Detector:
    def __init__(self, weights_path='../data/coco_weights.pth', iou_threshold=0.35, score_threshold=0.05):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        detector = YOLOv4()
        detector.load_state_dict(torch.load(weights_path, map_location=self.device))
        detector.to(self.device)
        self.detector = detector
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def set_iou_threshold(self, iou_threshold):
        assert 0 < iou_threshold < 1
        self.iou_threshold = iou_threshold

    def set_score_threshold(self, score_threshold):
        assert 0 < score_threshold < 1
        self.score_threshold = score_threshold

    def detect_image(self, filepath, size, iou_threshold=0.45, score_threshold=0.25):
        self.set_iou_threshold(iou_threshold)
        self.set_score_threshold(score_threshold)
        # read and prepare image
        preprocess_time = time.time()

        img = torchvision.io.read_image(filepath)
        img = torchvision.transforms.CenterCrop(size)(img)
        tensor_img = img / 255.
        tensor_img = tensor_img.view(1, *img.shape)
        tensor_img = tensor_img.cuda()
        tensor_img = torch.autograd.Variable(tensor_img)

        preprocess_time = time.time() - preprocess_time

        # get predictions
        detection_time = time.time()
        prediction = self.detector(tensor_img, size)

        prediction = non_maximum_suppression(prediction,
                                             iou_threshold=self.iou_threshold,
                                             score_threshold=self.score_threshold)
        detection_time = time.time() - detection_time
        print(f'Processing {os.path.basename(filepath)}')
        print('Preprocess time: {} ms'.format(int(preprocess_time * 1000)))
        print('Inference time: {} ms'.format(int(detection_time * 1000)), end='\n\n')
        # print('Total time: {preprocess_time + detection_time}'.format())
        prediction = prediction[0]
        box, prob, class_id = torch.split(prediction, [4, 1, 1], dim=-1)
        prob = prob.view(-1)
        class_id = class_id.view(-1).int()
        labels = [f'{COCO_NAMES[name_id]}' for name_id in class_id]
        # draw boxes
        detected_image = draw_bounding_boxes(img, box, labels, width=2, font_size=16)
        detected_image = detected_image.float() / 255.
        torchvision.utils.save_image(detected_image, os.path.basename(filepath))

