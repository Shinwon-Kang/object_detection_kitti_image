from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import torchvision

class KittiTestDataset(Dataset):
    
    def __init__(self, root_dir):
        super().__init__()

        self.root_dir = root_dir
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "image_2"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "image_2", self.imgs[idx])

        image = Image.open(img_path).convert("RGB")
        image = np.asarray(image).astype('float32') / 255.0

        image = torchvision.transforms.ToTensor()(image)

        return image

    def __len__(self):
        return len(self.imgs)
