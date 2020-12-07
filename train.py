import kitti_dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import visdom

# Run visdom
# python -m visdom.server


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def run():
    train_dataset = kitti_dataset.KittiDataset(is_train=True, 
                                        root_dir='/home/kangsinwon/3D_Object_Detection/KITTI_DATA/training')

    # DataLoader
    train_data_loader = DataLoader (
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    # norm = colors.LogNorm(image.mean() + 0.5 * image.std(), image.max(), clip='True')
    # plt.imshow(image, cmap=cm.gray, norm=norm, origin="lower")
    # images, targets = next(iter(train_data_loader))

    # To Cuda
    # images = list(image.to(device) for image in images)
    # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    # show_image(images, targets)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 9
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    num_epochs = 30

    # 2 - Modifying the model to add a different backbone
    # backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    # backbone.out_channels = 1280

    # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                aspect_ratios=((0.5, 1.0, 2.0),))

    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
    #                                                 output_size=7,
    #                                                 sampling_ratio=2)
    # model = FasterRCNN(backbone,
    #                 num_classes=2,
    #                 rpn_anchor_generator=anchor_generator,
    #                 box_roi_pool=roi_pooler)

    itr = 1
    model.to(device)

    # Visualize with visdom
    vis_loss = visdom.Visdom()
    vis_loss_iter = visdom.Visdom()

    plot_loss = vis_loss.line(Y=torch.tensor([0]), X=torch.tensor([0]))
    plot_loss_iter = vis_loss.line(Y=torch.tensor([0]), X=torch.tensor([0]))

    for epoch in range(num_epochs):
        for images, targets in train_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            # print('loss_dict', loss_dict)
            # print('targets', targets)

            print('loss: ', losses)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if not torch.isfinite(torch.as_tensor(loss_value)):
            	print('WARNING: non-finite loss, ending training ')
            	exit(1)

            if itr % 50 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")
                vis_loss_iter.line(Y=[loss_value], X=np.array([itr]), win=plot_loss_iter, update='append')

            itr += 1
            lr_scheduler.step()
        
        print(f"Epoch #{epoch} loss: {loss_value}")
        vis_loss.line(Y=[loss_value], X=np.array([epoch]), win=plot_loss, update='append')


    torch.save(model.state_dict(), 'model.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'ckpt.pth')


def show_image(images, targets):
    boxes = targets[0]['boxes'].cpu().numpy().astype(np.float32)
    image = images[0].permute(1, 2, 0).cpu().numpy()

    boxes[:, 0] = boxes[:, 0] * 1242
    boxes[:, 2] = boxes[:, 2] * 1242
    boxes[:, 1] = boxes[:, 1] * 375
    boxes[:, 3] = boxes[:, 3] * 375

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