from torch.utils import data
from torchvision import transforms
from PIL import Image
import torch
import glob
import numpy as np


class Portrait_dataset(data.Dataset):
    def __init__(self, img_paths, anno_paths, transform, label_transform):
        self.imgs = img_paths
        self.annos = anno_paths
        self.transform = transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        img = self.imgs[index]
        anno = self.annos[index]

        pil_img = Image.open(img)
        pil_img = pil_img.convert("RGB")
        img_tensor = self.transform(pil_img)

        pil_anno = Image.open(anno)
        anno_tensor = self.label_transform(pil_anno)
        anno_tensor = torch.squeeze(anno_tensor).type(torch.long)
        anno_tensor[anno_tensor > 0] = 1

        return img_tensor, anno_tensor

    def __len__(self):
        return len(self.imgs)


def imgpre(config):
    imgs = glob.glob("./dataset/img/*.png")
    imgs.sort()
    train_imgs = imgs[:500]
    test_imgs = imgs[500:]
    labels = glob.glob("./dataset/label/*.png")
    labels.sort()
    train_labels = labels[:500]
    test_labels = labels[500:]
    try:
        np.random.seed(config.get("seed"))
        index = np.random.permutation(len(train_imgs))

        train_imgs = np.array(train_imgs)[index]
        train_labels = np.array(train_labels)[index]

        transform = transforms.Compose([
            transforms.Resize((config.get("IMG_SIZE"), config.get("IMG_SIZE"))),
            transforms.CenterCrop((config.get("SUO_FANG_IMG_SIZE"), config.get("SUO_FANG_IMG_SIZE"))),
            transforms.ToTensor(),
        ])

        label_transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((config.get("IMG_SIZE"), config.get("IMG_SIZE"))),
            transforms.CenterCrop((config.get("SUO_FANG_IMG_SIZE"), config.get("SUO_FANG_IMG_SIZE"))),
            transforms.ToTensor(),
        ])

        train_dataset = Portrait_dataset(train_imgs, train_labels, transform, label_transform)
        test_dataset = Portrait_dataset(test_imgs, test_labels, transform, label_transform)
        train_dl = data.DataLoader(
            train_dataset,
            batch_size=config.get("batch_size"),
            shuffle=True,
        )
        test_dl = data.DataLoader(
            test_dataset,
            batch_size=config.get("batch_size"),
        )
        return train_dl, test_dl
    except Exception as e:
        print(e)
