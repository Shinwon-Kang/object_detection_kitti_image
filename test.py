import kitti_dataset
import kitti_test_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import visdom
from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def run():
    test_dataset = kitti_test_dataset.KittiTestDataset(root_dir='/home/kangsinwon/3D_Object_Detection/KITTI_DATA/testing')

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 9
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load('model.pth'))
    
    model.to(device)
    model.eval()

    img = test_dataset[0]
    with torch.no_grad():
        pre = model([img.to(device)])
        print(pre)
    # show_image(images, outputs)
    # print(outputs[0]['labels'])


def show_image(images, targets):
    boxes = targets[0]['boxes'].cpu().numpy().astype(np.float32)
    image = images[0].permute(1, 2, 0).cpu().numpy()

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_xlabel('Normalize')
    ax1.imshow(image)
    for box in boxes:
        ax1.add_patch(
            patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=1,
                edgecolor='red',
                fill=False
            )
        )

    plt.show()


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    run()