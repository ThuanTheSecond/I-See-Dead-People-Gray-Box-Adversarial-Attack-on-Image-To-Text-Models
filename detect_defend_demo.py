import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import io
import numpy as np
import clip
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import scipy.ndimage as ndimage
import argparse
from utils_detect_defend import load_git_model, get_caption_from_csv, load_dataset
from utils import predict

class ClipTransformDefender:
    def __init__(self, clip_model, model, model_name, tokenizer, image_processor, 
                 detection_threshold=0.85, transform_var_threshold=0.08, transform_mean_threshold=0.75,
                 device='cuda', use_csv=True, use_git=False):
        """
        Initialize ClipTransformDefender with CLIP and captioning models.
        
        Args:
            clip_model: CLIP model for encoding images and text.
            model: Captioning model (e.g., ViT-GPT2 or BLIP).
            model_name: Name of the captioning model ('vit-gpt2' or 'blip').
            tokenizer: Tokenizer for the captioning model.
            image_processor: Image processor for the captioning model.
            detection_threshold: Cosine similarity threshold for semantic consistency (default: 0.85).
            transform_var_threshold: Variance threshold for transform stability (default: 0.08).
            transform_mean_threshold: Mean threshold for transform stability (default: 0.75).
            device: Device to run models ('cuda' or 'cpu').
            use_csv: Whether to read captions from CSV file (default: True).
            use_git: Whether to use GIT model for generating temporary captions (default: False).
        """
        self.clip_model = clip_model
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        self.use_csv = use_csv
        self.use_git = use_git
        self.csv_path = "/kaggle/input/flickr30k/captions.txt" if use_csv else None
        self.transformations = [
            ('original', lambda x: x),
            ('jpeg_75', lambda x: self._jpeg_compress(x, 75)),
            ('jpeg_90', lambda x: self._jpeg_compress(x, 90)),
            ('gaussian_blur', transforms.GaussianBlur(kernel_size=3)),
            ('median_blur', self._median_filter),
            ('bit_depth', self._bit_depth_reduction),
            ('color_jitter', transforms.ColorJitter(brightness=0.1, contrast=0.1)),
        ]
        self.threshold = detection_threshold
        self.transform_var_threshold = transform_var_threshold
        self.transform_mean_threshold = transform_mean_threshold
        
        # Initialize GIT model if use_git is True
        if self.use_git:
            self.git_processor, self.git_model = load_git_model(device)

    def _median_filter(self, img, kernel_size=3):
        """Apply median filter using scipy.ndimage."""
        np_img = img.cpu().numpy()
        if len(np_img.shape) == 3:
            np_img = np_img.transpose(1, 2, 0)
        filtered = ndimage.median_filter(np_img, size=kernel_size)
        if len(filtered.shape) == 3 and filtered.shape[2] == 3:
            filtered = filtered.transpose(2, 0, 1)
        return torch.from_numpy(filtered).to(self.device)

    def _jpeg_compress(self, img, quality=90):
        """Apply JPEG compression with specified quality."""
        img_pil = TF.to_pil_image((img + 1)/2)
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return TF.to_tensor(Image.open(buffer)).to(self.device) * 2 - 1

    def _bit_depth_reduction(self, img, bits=5):
        """Reduce bit depth of the image."""
        img = torch.clamp(img, -1, 1)
        max_val = 2**bits - 1
        img_scaled = ((img + 1) / 2 * max_val).round() / max_val
        return img_scaled * 2 - 1

    def detect_attack(self, image, caption):
        """Detect adversarial attack by checking semantic consistency and transform stability."""
        try:
            image_resized = TF.resize(image, (224, 224), antialias=True)
            with torch.no_grad():
                # Check if image is batched (4D) or unbatched (3D)
                if image_resized.dim() == 3:
                    image_features = self.clip_model.encode_image(image_resized.unsqueeze(0))
                else:
                    image_features = self.clip_model.encode_image(image_resized)
                
                text_tokens = clip.tokenize([caption]).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                similarity = F.cosine_similarity(image_features, text_features).item()
                
                transform_similarities = []
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(lambda t: t[1](image.clone()), t) 
                             for t in self.transformations[1:]]
                    for future in as_completed(futures):
                        try:
                            transformed = future.result()
                            transformed_resized = TF.resize(transformed, (224, 224), antialias=True)
                            if transformed_resized.dim() == 3:
                                transformed_features = self.clip_model.encode_image(transformed_resized.unsqueeze(0))
                            else:
                                transformed_features = self.clip_model.encode_image(transformed_resized)
                            transform_sim = F.cosine_similarity(image_features, transformed_features).item()
                            transform_similarities.append(transform_sim)
                        except Exception as e:
                            print(f"Transform error: {str(e)[:100]}")
                            continue
                
                transform_variance = np.var(transform_similarities) if transform_similarities else 0
                transform_mean = np.mean(transform_similarities) if transform_similarities else 0
                
                is_semantic_inconsistent = similarity < self.threshold
                is_unstable_to_transforms = (transform_variance > self.transform_var_threshold or 
                                           transform_mean < self.transform_mean_threshold)
                
                if is_semantic_inconsistent and is_unstable_to_transforms:
                    return True, 0.7 + (1.0 - similarity) * 0.3, "targeted" if similarity < 0.6 else "untargeted"
                elif is_semantic_inconsistent:
                    return True, 0.6 + (1.0 - similarity) * 0.2, "unknown"
                else:
                    return False, 0.0, "none"
                    
        except Exception as e:
            print(f"Detection error: {str(e)[:100]}")
            return False, 0.0, "error"

    def defend(self, image, image_id=None, original_caption=None):
        """
        Defend against adversarial attack by generating a reliable caption.
        
        Args:
            image: Input image tensor.
            image_id: Image identifier (e.g., filename) to lookup caption in CSV (optional).
            original_caption: Reference caption (optional).
        
        Returns:
            caption: Final caption (original or defended).
            confidence: Confidence score.
            defense_info: Dictionary with detection and defense details.
        """
        try:
            if original_caption is None and image_id is not None and self.use_csv:
                # Lookup caption from CSV if enabled and image_id provided
                original_caption = get_caption_from_csv(self.csv_path, image_id)
            
            if original_caption is None and self.use_git:
                # Generate temporary caption using GIT if enabled
                inputs = self.git_processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.git_model.generate(**inputs, max_length=16)
                original_caption = self.git_processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            if original_caption is None:
                # Fallback: assume attack and proceed to defense
                is_adversarial, confidence, attack_type = True, 0.5, "unknown"
            else:
                is_adversarial, confidence, attack_type = self.detect_attack(image, original_caption)
            
            defense_info = {
                'is_adversarial': is_adversarial,
                'detection_confidence': confidence,
                'attack_type': attack_type,
                'transformations': []
            }
            
            if not is_adversarial and original_caption:
                print(f"Image is clean, returning original caption: {original_caption}")
                return original_caption, 1.0, defense_info
            
            # Defense: generate captions from transformed images
            captions = []
            similarities = []
            with ThreadPoolExecutor() as executor:
                futures = []
                for name, transform in self.transformations:
                    img_copy = image.clone()
                    futures.append(executor.submit(
                        self._apply_transform_and_predict,
                        img_copy, name, transform
                    ))
                
                for future in as_completed(futures):
                    try:
                        name, caption, similarity = future.result()
                        captions.append(caption)
                        similarities.append(similarity)
                        defense_info['transformations'].append({
                            'name': name,
                            'caption': caption,
                            'similarity': similarity
                        })
                    except Exception as e:
                        print(f"Defense transform error ({name}): {str(e)[:100]}")
                        continue
            
            if not captions:
                print("No captions generated, returning original or empty caption")
                return original_caption if original_caption else "", 0.0, defense_info
            
            # Select best caption based on attack type
            if attack_type == "targeted":
                caption_counter = Counter(captions)
                if caption_counter:
                    best_caption, count = caption_counter.most_common(1)[0]
                    confidence = max(similarities) if count/len(captions) < 0.4 else count/len(captions)
                else:
                    best_idx = np.argmax(similarities)
                    best_caption, confidence = captions[best_idx], similarities[best_idx]
            else:
                best_idx = np.argmax(similarities)
                best_caption, confidence = captions[best_idx], similarities[best_idx]
            
            defense_info['defense_method'] = 'ensemble_voting' if attack_type == 'targeted' else 'best_similarity'
            defense_info['defense_confidence'] = confidence
            print(f"Defended caption: {best_caption} with confidence: {confidence}")
            
            return best_caption, confidence, defense_info
            
        except Exception as e:
            print(f"Defense error: {str(e)[:100]}")
            return original_caption if original_caption else "", 0.0, {'error': str(e)}

    def _apply_transform_and_predict(self, image, name, transform):
        """Apply transformation and predict caption."""
        try: 
            transformed = transform(image)
            caption = predict(
                self.model_name, 
                self.model, 
                self.tokenizer, 
                self.image_processor, 
                transformed.unsqueeze(0)
            )[0]
            transformed_resized = TF.resize(transformed, (224, 224), antialias=True)
            with torch.no_grad():
                if transformed_resized.dim() == 3:
                    image_features = self.clip_model.encode_image(transformed_resized.unsqueeze(0))
                else:
                    image_features = self.clip_model.encode_image(transformed_resized)
                text_tokens = clip.tokenize([caption]).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                similarity = F.cosine_similarity(image_features, text_features).item()
            
            return name, caption, similarity
        except Exception as e:
            print(f"Transform {name} error: {str(e)[:100]}")
            raise e

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Image-to-Text Adversarial Defense")
    parser.add_argument("--model", type=str, default="vit-gpt2", choices=["vit-gpt2", "blip"],
                        help="Captioning model name (vit-gpt2 or blip)")
    parser.add_argument("--dataset", type=str, default="flickr30k", 
                        choices=["flickr30k", "targeted_attack", "untargeted_attack"],
                        help="Dataset to use (flickr30k, targeted_attack, or untargeted_attack)")
    parser.add_argument("--image_path", type=str, 
                        help="Path to input image (optional if dataset is used)")
    parser.add_argument("--image_id", type=str, 
                        help="Image identifier (e.g., filename) for CSV lookup")
    parser.add_argument("--caption", type=str, 
                        help="Original caption (optional, overrides CSV lookup)")
    parser.add_argument("--use_csv", action="store_true", default=True,
                        help="Read captions from CSV file (default: True)")
    parser.add_argument("--use_git", action="store_true",
                        help="Use GIT model to generate temporary caption if no caption provided")
    parser.add_argument("--num_images", type=int, default=1,
                        help="Number of images to process from dataset (default: 1)")
    args = parser.parse_args()

    # Load CLIP model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load captioning model
    from utils import load_model
    image_processor, tokenizer, model, _, _, _ = load_model(args.model)
    
    # Initialize defender
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    defender = ClipTransformDefender(
        clip_model=clip_model,
        model=model,
        model_name=args.model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        device=device,
        use_csv=args.use_csv,
        use_git=args.use_git
    )
    
    # Load dataset
    dataloader = load_dataset(args.dataset, image_processor, batch_size=1, num_images=args.num_images)
    
    # Process images
    if args.image_path:
        # Process single image from path
        image = Image.open(args.image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).to(device)
        caption, confidence, defense_info = defender.defend(
            image=image,
            image_id=args.image_id,
            original_caption=args.caption
        )
        print(f"Image: {args.image_path}")
        print(f"Final caption: {caption}")
        print(f"Confidence: {confidence}")
        print(f"Defense info: {defense_info}")
    else:
        # Process images from dataset
        for i, batch in enumerate(dataloader):
            if i >= args.num_images:
                break
            image = batch['image'].to(device)
            caption = batch['caption'][0] if args.caption is None else args.caption
            image_id = batch['image_id'][0] if args.image_id is None else args.image_id
            print(f"\nProcessing image {i+1}/{args.num_images} (ID: {image_id})")
            caption, confidence, defense_info = defender.defend(
                image=image,
                image_id=image_id,
                original_caption=caption
            )
            print(f"Final caption: {caption}")
            print(f"Confidence: {confidence}")
            print(f"Defense info: {defense_info}")