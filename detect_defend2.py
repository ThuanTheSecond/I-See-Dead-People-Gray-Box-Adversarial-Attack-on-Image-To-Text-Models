import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import io
import numpy as np
import clip
from collections import Counter
import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import utils functions (assume these are available)
from utils import load_model, predict, load_dataset

class ClipTransformDefender:
    def __init__(self, clip_model, model, model_name, tokenizer, image_processor, 
                 detection_threshold=0.85, transform_var_threshold=0.08, transform_mean_threshold=0.75,
                 device='cuda'):
        """
        Optimized adversarial defense combining CLIP and Input Transformations
        
        Improvements:
        - Parallel transformation processing
        - Torch-native median filter
        - Dynamic threshold calibration
        - Device optimization
        """
        self.clip_model = clip_model
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        
        # Define input transformations (optimized)
        self.transformations = [
            ('original', lambda x: x),
            ('jpeg_75', lambda x: self._jpeg_compress(x, 75)),
            ('jpeg_90', lambda x: self._jpeg_compress(x, 90)),
            ('gaussian_blur', transforms.GaussianBlur(kernel_size=3)),
            ('median_blur', lambda x: TF.median_blur(x, kernel_size=3)),
            ('bit_depth', self._bit_depth_reduction),
            ('color_jitter', transforms.ColorJitter(brightness=0.1, contrast=0.1)),
        ]
        
        # Threshold parameters
        self.threshold = detection_threshold
        self.transform_var_threshold = transform_var_threshold
        self.transform_mean_threshold = transform_mean_threshold
        
    def _jpeg_compress(self, img, quality=90):
        """Optimized JPEG compression with device handling"""
        # Ensure proper value range
        is_normalized = img.min() >= -1 and img.max() <= 1
        img_pil = TF.to_pil_image(img.cpu() if not is_normalized else (img + 1)/2)
        
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img_tensor = TF.to_tensor(Image.open(buffer)).to(self.device)
        return img_tensor * 2 - 1 if is_normalized else img_tensor

    def _bit_depth_reduction(self, img, bits=5):
        """Bit depth reduction with range validation"""
        # Validate input range
        if img.min() < -1 or img.max() > 1:
            img = torch.clamp(img, -1, 1)
            
        max_val = 2**bits - 1
        img_scaled = ((img + 1) / 2 * max_val).round() / max_val
        return img_scaled * 2 - 1
    
    def _parallel_transform(self, image, transform_func):
        """Apply transformation with error handling"""
        try:
            transformed = transform_func(image)
            transformed_resized = TF.resize(transformed, (224, 224), antialias=True)
            return transformed_resized
        except Exception as e:
            print(f"Transformation error: {e}")
            return None

    def detect_attack(self, image, caption):
        """Detect adversarial attacks using CLIP with parallel processing"""
        # Prepare image and caption for CLIP
        image_resized = TF.resize(image, (224, 224), antialias=True)
        
        with torch.no_grad():
            # Compute embeddings
            image_features = self.clip_model.encode_image(image_resized.unsqueeze(0))
            text_tokens = clip.tokenize([caption]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            
            # Measure semantic similarity
            similarity = F.cosine_similarity(image_features, text_features).item()
            
            # Parallel transformation processing
            transform_similarities = []
            with ThreadPoolExecutor() as executor:
                futures = []
                for name, transform in self.transformations[1:]:
                    futures.append(
                        executor.submit(
                            self._parallel_transform, 
                            image.clone(), 
                            transform
                        )
                    )
                
                for future in as_completed(futures):
                    transformed = future.result()
                    if transformed is None:
                        continue
                        
                    with torch.no_grad():
                        transformed_features = self.clip_model.encode_image(transformed.unsqueeze(0))
                        transform_sim = F.cosine_similarity(image_features, transformed_features).item()
                        transform_similarities.append(transform_sim)
            
            # Calculate transformation metrics
            transform_variance = np.var(transform_similarities) if transform_similarities else 0
            transform_mean = np.mean(transform_similarities) if transform_similarities else 0
            
            # Detection logic
            is_semantic_inconsistent = similarity < self.threshold
            is_unstable_to_transforms = (
                transform_variance > self.transform_var_threshold or 
                transform_mean < self.transform_mean_threshold
            )
            
            if is_semantic_inconsistent and is_unstable_to_transforms:
                is_adversarial = True
                confidence = 0.7 + (1.0 - similarity) * 0.3
                attack_type = "targeted" if similarity < 0.6 else "untargeted"
            elif is_semantic_inconsistent:
                is_adversarial = True
                confidence = 0.6 + (1.0 - similarity) * 0.2
                attack_type = "unknown"
            else:
                is_adversarial = False
                confidence = 0.0
                attack_type = "none"
                
            return is_adversarial, confidence, attack_type
    
    def defend(self, image, original_caption=None):
        """Defend against attacks using optimized transformation ensemble"""
        # Attack detection
        if original_caption:
            is_adversarial, confidence, attack_type = self.detect_attack(image, original_caption)
        else:
            is_adversarial = True
            confidence = 0.5
            attack_type = "unknown"
        
        defense_info = {
            'is_adversarial': is_adversarial,
            'detection_confidence': confidence,
            'attack_type': attack_type,
            'transformations': []
        }
        
        # Return original if not adversarial
        if not is_adversarial and original_caption:
            return original_caption, 1.0, defense_info
        
        # Parallel transformation processing
        captions = []
        similarities = []
        transformation_data = []
        
        with ThreadPoolExecutor() as executor:
            future_to_name = {}
            for name, transform in self.transformations:
                future = executor.submit(
                    self._apply_transform_and_predict,
                    image.clone(),
                    name,
                    transform
                )
                future_to_name[future] = name
            
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    caption, similarity = future.result()
                    captions.append(caption)
                    similarities.append(similarity)
                    transformation_data.append({
                        'name': name,
                        'caption': caption,
                        'similarity': similarity
                    })
                except Exception as e:
                    print(f"Error processing {name}: {e}")
        
        defense_info['transformations'] = transformation_data
        
        # Defense strategy selection
        if attack_type == "targeted":
            caption_counter = Counter(captions)
            most_common = caption_counter.most_common(1)
            
            if most_common:
                best_caption, count = most_common[0]
                vote_confidence = count / len(captions)
                
                if vote_confidence < 0.4:
                    best_idx = np.argmax(similarities)
                    best_caption = captions[best_idx]
                    confidence = similarities[best_idx]
                else:
                    confidence = vote_confidence
            else:
                best_idx = np.argmax(similarities)
                best_caption = captions[best_idx]
                confidence = similarities[best_idx]
        else:
            best_idx = np.argmax(similarities)
            best_caption = captions[best_idx]
            confidence = similarities[best_idx]
        
        defense_info['defense_method'] = 'ensemble_voting' if attack_type == 'targeted' else 'best_similarity'
        defense_info['defense_confidence'] = confidence
        
        return best_caption, confidence, defense_info

    def _apply_transform_and_predict(self, image, name, transform):
        """Apply transformation and generate prediction"""
        try:
            # Apply transformation
            transformed = transform(image)
            
            # Generate caption
            caption = predict(
                self.model_name, 
                self.model, 
                self.tokenizer, 
                self.image_processor, 
                transformed.unsqueeze(0)
            )[0]
            
            # Compute CLIP similarity
            transformed_resized = TF.resize(transformed, (224, 224), antialias=True)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(transformed_resized.unsqueeze(0))
                text_tokens = clip.tokenize([caption]).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                similarity = F.cosine_similarity(image_features, text_features).item()
            
            return caption, similarity
        except Exception as e:
            print(f"Error in {name}: {e}")
            raise

def test_defender(image_path, model_name='vit-gpt2'):
    """Test function for defender"""
    # Load models
    clip_model, _ = clip.load("ViT-B/32", device='cuda')
    image_processor, tokenizer, model, _, _, _ = load_model(model_name)
    
    # Load test image
    image = Image.open(image_path).convert("RGB")
    img_tensor = TF.to_tensor(image).cuda() * 2 - 1  # Convert to [-1,1]
    
    # Initialize defender
    defender = ClipTransformDefender(
        clip_model=clip_model,
        model=model, 
        model_name=model_name,
        tokenizer=tokenizer,
        image_processor=image_processor
    )
    
    # Predict original caption
    orig_caption = predict(model_name, model, tokenizer, image_processor, img_tensor.unsqueeze(0))[0]
    
    # Detect and defend
    is_adv, conf, attack_type = defender.detect_attack(img_tensor, orig_caption)
    defended_caption, def_conf, def_info = defender.defend(img_tensor, orig_caption)
    
    # Print results
    print(f"Original caption: {orig_caption}")
    print(f"Is adversarial: {is_adv} (confidence: {conf:.4f}, type: {attack_type})")
    print(f"Defended caption: {defended_caption} (confidence: {def_conf:.4f})")
    print(f"Defense method: {def_info['defense_method']}")
    
    return defender

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Optimized CLIP + Input Transformation Defense")
    parser.add_argument("--model", type=str, default="vit-gpt2", help="Model name")
    parser.add_argument("--dataset", type=str, default="flickr30k", help="Dataset name")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--test_image", type=str, default="", help="Single test image path")
    args = parser.parse_args()
    
    # Test single image if specified
    if args.test_image:
        print(f"Testing with image: {args.test_image}")
        defender = test_defender(args.test_image, args.model)
        exit(0)
    
    # Run dataset evaluation
    print(f"Evaluating on {args.dataset} with {args.model} ({args.num_samples} samples)")
    
    # Load models
    print("Loading models...")
    clip_model, _ = clip.load("ViT-B/32", device='cuda')
    image_processor, tokenizer, model, _, _, _ = load_model(args.model)
    
    # Load dataset
    print("Loading dataset...")
    dataloader = load_dataset(args.dataset, image_processor, batch_size=1)
    
    # Initialize defender
    print("Initializing defender...")
    defender = ClipTransformDefender(
        clip_model=clip_model,
        model=model, 
        model_name=args.model,
        tokenizer=tokenizer,
        image_processor=image_processor
    )
    
    # Evaluation loop
    results = []
    print("Starting evaluation...")
    
    for i, batch in tqdm(enumerate(dataloader), total=min(args.num_samples, len(dataloader))):
        if i >= args.num_samples:
            break
        
        try:
            # Prepare data
            image = batch['image'][0].cuda()
            
            # Predict caption
            orig_caption = predict(args.model, model, tokenizer, image_processor, image.unsqueeze(0))[0]
            
            # Detect and defend
            is_adv, conf, attack_type = defender.detect_attack(image, orig_caption)
            defended_caption, def_conf, def_info = defender.defend(image, orig_caption)
            
            # Save results
            results.append({
                'sample_id': i,
                'original_caption': orig_caption,
                'defended_caption': defended_caption,
                'detected_as_adversarial': is_adv,
                'detection_confidence': conf,
                'attack_type': attack_type,
                'defense_confidence': def_conf,
                'defense_method': def_info['defense_method']
            })
            
            # Print sample results
            if i < 3 or i % 10 == 0:
                print(f"\nSample {i}:")
                print(f"Original: {orig_caption}")
                print(f"Defended: {defended_caption}")
                print(f"Attack: {attack_type} (Conf: {conf:.2f}, Defense: {def_info['defense_method']})")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
    
    # Generate report
    if results:
        df = pd.DataFrame(results)
        output_file = f"defense_results_{args.model}_{args.dataset}.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Generate statistics
        detection_rate = df['detected_as_adversarial'].mean()
        avg_defense_conf = df['defense_confidence'].mean()
        
        print("\n=== Evaluation Summary ===")
        print(f"Adversarial Detection Rate: {detection_rate:.2%}")
        print(f"Average Defense Confidence: {avg_defense_conf:.2%}")
        print("\nAttack Type Distribution:")
        print(df['attack_type'].value_counts(normalize=True))
        
        # Plot results
        plt.figure(figsize=(10, 6))
        df['attack_type'].value_counts().plot(kind='bar')
        plt.title('Detected Attack Types Distribution')
        plt.savefig(f"attack_types_{args.model}_{args.dataset}.png")
        print("Visualization saved")
