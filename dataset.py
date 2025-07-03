import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import torch
import json
import cv2

class ImageNet(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open("/cs_storage/public_datasets/imagenet_class_index.json", "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        # with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
        #             self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            # x.resize((224, 224))
            if self.transform:
                # image_processor(images=images, return_tensors="pt").pixel_values.cuda()
                x = self.transform(images=x, return_tensors="pt").pixel_values.squeeze(0)
            return x, self.targets[idx]

class Flickr30k(Dataset):
    def __init__(self, image_filenames, captions, transform):


        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.transform = transform

    def __getitem__(self, idx):
        item = {}
        image = Image.open(f"flickr30k/Images/{self.image_filenames[idx]}").convert("RGB")
        if self.transform:
            image = self.transform(images=image, return_tensors="pt").pixel_values.squeeze(0)
        item['image'] = image.float()
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)
        
if __name__ == '__main__':
    print('Test1')
    train_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )
    print('Test2')
    dataset = Flickr30k("/cs_storage/public_datasets/flickr30k-images", "/cs_storage/public_datasets/flickr30k-labels", train_transform)
    print('Hello')
