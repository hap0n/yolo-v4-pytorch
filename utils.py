import torch
from torch import nn
from torchvision.ops import nms
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import random
import numpy as np
import matplotlib.patches as patches
from PIL import Image
import os


class Mish(nn.Module):
    """ Mish activation module """
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x): # noqa
        return x * (torch.tanh(nn.Softplus()(x)))


class Conv(nn.Module):
    """ Convolutional layer with batch normalization and activation layer (LeakyReLU or Mish) """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 activation=nn.LeakyReLU(0.1), use_bn=True, bias=False):
        super(Conv, self).__init__()
        padding = (kernel_size - 1) // 2

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        if use_bn:
            self.layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            self.layers.append(activation)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def non_maximum_suppression(prediction, iou_threshold=0.45, score_threshold=0.25):
    """ Performs non-maximum suppression on the boxes according to their intersection-over-union

    Args:
        prediction: (Tensor[batch_size,N,85]) boxes to perform NMS on, confidences for each one of the classes,
            class probabilities for each one of the boxes in format [4, 1, 80 (for COCO dataset)]
        iou_threshold: (float) discards all overlapping boxes with IoU > iou_threshold
        score_threshold: (float) discards all boxes with score < score_threshold

    Returns:
        boxes_keep: (Tensor[M,4]) boxes after NMS done
        labels: (List[M]) list of labels with length of M. Example of label sample: "dog 0.979"
    """

    # num_classes = len(names)
    max_wh = 4096
    max_det = 300
    max_nms = 30000
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    for xi, x in enumerate(prediction):
        x = x[x[..., 4] > score_threshold]

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :4]

        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > score_threshold]

        # Filter by class
        # if classes is not None:
        #     x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = nms(boxes, scores, iou_threshold)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]

    return output


def draw_and_save_output_image(image_path, detections, img_size, output_path, classes):
    # Create plot
    img = np.array(Image.open(image_path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    # Rescale boxes to original image
    # detections = rescale_boxes(detections, img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
    bbox_colors = random.sample(colors, n_cls_preds)
    for x1, y1, x2, y2, conf, cls_pred in detections:

        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

        box_w = x2 - x1
        box_h = y2 - y1

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        plt.text(
            x1,
            y1,
            s=classes[int(cls_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0})

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = os.path.basename(image_path).split(".")[0]
    output_path = os.path.join(output_path, f"{filename}.png")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def rescale_boxes(boxes, current_dim, original_shape):
    """
    Rescales bounding boxes to the original shape
    """
    orig_h, orig_w = original_shape

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes
