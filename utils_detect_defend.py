import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from torch.utils.data import DataLoader, Subset
from dataset import Flickr30k

def load_git_model(device='cuda'):
    try:
        processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco").to(device).eval()
        return processor, model
    except Exception as e:
        print(f"Error loading GIT model: {str(e)[:100]}")
        return None, None

def get_caption_from_csv(csv_path, image_id):
    try:
        df = pd.read_csv(csv_path)
        caption = df[df['image'] == image_id]['caption'].values
        if len(caption) > 0:
            return caption[0]
        else:
            print(f"No caption found for image ID: {image_id}")
            return None
    except Exception as e:
        print(f"Error reading CSV: {str(e)[:100]}")
        return None

def load_dataset(dataset, image_processor, batch_size=1, num_images=1):
    try:
        if dataset == 'flickr30k':
            csv_path = "/kaggle/input/flickr30k/captions.txt"
            df = pd.read_csv(csv_path)
            
            # Lấy unique images và caption đầu tiên
            unique_images = df.groupby('image').first().reset_index()
            print(f"Total unique images: {len(unique_images)}")
            print(f"Requested images: {num_images}")
            
            # Giới hạn số lượng ảnh
            unique_images = unique_images.iloc[:min(num_images, len(unique_images))]
            
            dataset = Flickr30k(
                unique_images["image"].values,
                unique_images["caption"].values,
                transform=image_processor,
            )
        elif dataset == 'targeted_attack':
            csv_path = "/path/to/targeted_attack_captions.csv"
            df = pd.read_csv(csv_path)
            unique_images = df.groupby('image').first().reset_index()
            unique_images = unique_images.iloc[:min(num_images, len(unique_images))]
            dataset = Flickr30k(
                unique_images["image"].values,
                unique_images["caption"].values,
                transform=image_processor,
            )
        elif dataset == 'untargeted_attack':
            csv_path = "/path/to/untargeted_attack_captions.csv"
            df = pd.read_csv(csv_path)
            unique_images = df.groupby('image').first().reset_index()
            unique_images = unique_images.iloc[:min(num_images, len(unique_images))]
            dataset = Flickr30k(
                unique_images["image"].values,
                unique_images["caption"].values,
                transform=image_processor,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        def collate_fn(batch):
            images = torch.stack([item['image'] for item in batch])
            captions = [item['caption'] for item in batch]
            image_ids = [item['image_id'] for item in batch]
            return {
                'image': images.squeeze(1) if images.dim() == 5 else images,
                'caption': captions,
                'image_id': image_ids
            }
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            collate_fn=collate_fn
        )
        return dataloader
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return None