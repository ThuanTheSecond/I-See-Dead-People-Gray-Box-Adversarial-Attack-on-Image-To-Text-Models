import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from torch.utils.data import DataLoader
from dataset import Flickr30k
from torch.utils.data import Subset

def load_git_model(device='cuda'):
    """
    Load the GIT model and processor for caption generation.
    
    Args:
        device: Device to run the model ('cuda' or 'cpu').
    
    Returns:
        processor: GIT processor.
        model: GIT model.
    """
    try:
        processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
        model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco").to(device).eval()
        return processor, model
    except Exception as e:
        print(f"Error loading GIT model: {str(e)[:100]}")
        return None, None

def get_caption_from_csv(csv_path, image_id):
    """
    Retrieve caption from CSV file based on image ID.
    
    Args:
        csv_path: Path to CSV file with image-caption pairs.
        image_id: Image identifier (e.g., filename).
    
    Returns:
        caption: Corresponding caption or None if not found.
    """
    try:
        df = pd.read_csv(csv_path)
        caption = df[df['image'] == image_id]['caption'].values
        if len(caption) > 0:
            return caption[0]  # Return first caption if multiple exist
        else:
            print(f"No caption found for image ID: {image_id}")
            return None
    except Exception as e:
        print(f"Error reading CSV: {str(e)[:100]}")
        return None

def load_dataset(dataset, image_processor, batch_size=1, num_images=1):
    """
    Load dataset for evaluation, limited to specified number of images.
    
    Args:
        dataset: Dataset name ('flickr30k', 'targeted_attack', or 'untargeted_attack').
        image_processor: Image processor for the captioning model.
        batch_size: Batch size for DataLoader (default: 1).
        num_images: Number of images to load from dataset (default: 1).
    
    Returns:
        dataloader: DataLoader for the specified dataset.
    """
    try:
        if dataset == 'flickr30k':
            csv_path = "/kaggle/input/flickr30k/captions.txt"
            df = pd.read_csv(csv_path)
            dataset = Flickr30k(
                df["image"].values,
                df["caption"].values,
                transform=image_processor,
            )
        elif dataset == 'targeted_attack':
            # Placeholder: Assume CSV with targeted attack images and captions
            csv_path = "/path/to/targeted_attack_captions.csv"  # Update with actual path
            df = pd.read_csv(csv_path)
            dataset = Flickr30k(
                df["image"].values,
                df["caption"].values,
                transform=image_processor,
            )
        elif dataset == 'untargeted_attack':
            # Placeholder: Assume CSV with untargeted attack images and captions
            csv_path = "/path/to/untargeted_attack_captions.csv"  # Update with actual path
            df = pd.read_csv(csv_path)
            dataset = Flickr30k(
                df["image"].values,
                df["caption"].values,
                transform=image_processor,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Limit dataset to num_images
        indices = list(range(min(num_images, len(dataset))))
        dataset = Subset(dataset, indices)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1,
            shuffle=False
        )
        return dataloader
    except Exception as e:
        print(f"Error loading dataset: {str(e)[:100]}")
        return None