import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
import argparse
import clip
from utils import load_model, load_dataset, predict

class RandomizedSmoothingDefense:
    def __init__(self, model, model_name, tokenizer, image_processor, 
                 noise_std=0.5, num_samples=100, confidence_threshold=0.7, batch_size=16):
        """
        Randomized Smoothing Defense for Image-to-Text models - GPU Optimized
        
        Args:
            model: The image-to-text model (ViT-GPT2 or BLIP)
            model_name: Name of the model
            tokenizer: Model tokenizer
            image_processor: Image processor
            noise_std: Standard deviation of Gaussian noise
            num_samples: Number of noisy samples to generate
            confidence_threshold: Threshold for detection
            batch_size: Batch size for GPU processing
        """
        self.model = model.eval()  # Ensure model is in eval mode
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.noise_std = noise_std
        self.num_samples = num_samples
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def add_gaussian_noise_batch(self, image, std, num_samples):
        """Add Gaussian noise to create batch of noisy images - GPU optimized"""
        # Create batch of the same image
        batch_images = image.unsqueeze(0).repeat(num_samples, 1, 1, 1)
        # Generate noise for entire batch
        noise = torch.randn_like(batch_images, device=self.device) * std
        noisy_batch = batch_images + noise
        return torch.clamp(noisy_batch, -1, 1)
    
    def batch_predict(self, image_batch):
        """Batch prediction for multiple images - GPU optimized"""
        all_captions = []
        
        # Process in smaller batches to avoid memory issues
        for i in range(0, len(image_batch), self.batch_size):
            batch_slice = image_batch[i:i+self.batch_size]
            
            with torch.no_grad():  # Save GPU memory
                if self.model_name == 'vit-gpt2':
                    # Use the encoder-decoder structure efficiently
                    pixel_values = batch_slice
                    generated_ids = self.model.generate(
                        pixel_values,
                        max_length=50,
                        num_beams=1,  # Reduce beams for speed
                        do_sample=False,
                        early_stopping=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    captions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    
                elif self.model_name == 'blip':
                    # BLIP batch processing
                    generated_ids = self.model.generate(
                        batch_slice,
                        max_length=50,
                        num_beams=1,
                        do_sample=False,
                        early_stopping=True
                    )
                    captions = self.model.processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                all_captions.extend(captions)
        
        return all_captions
    
    def smooth_predict_optimized(self, image):
        """
        GPU-optimized randomized smoothing prediction
        
        Args:
            image: Input image tensor
            
        Returns:
            smoothed_caption: Most frequent caption
            confidence: Confidence score (0-1)
            caption_distribution: Distribution of captions
        """
        # Generate all noisy images at once
        noisy_batch = self.add_gaussian_noise_batch(image, self.noise_std, self.num_samples)
        
        # Batch prediction - much faster on GPU
        captions = self.batch_predict(noisy_batch)
        
        # Count caption frequencies (this part still on CPU but much less overhead)
        caption_counter = Counter(captions)
        most_common_caption, most_common_count = caption_counter.most_common(1)[0]
        
        # Calculate confidence as frequency of most common caption
        confidence = most_common_count / self.num_samples
        
        return most_common_caption, confidence, caption_counter
    
    def detect_adversarial(self, image, original_caption=None):
        """
        GPU-optimized adversarial detection
        """
        smoothed_caption, stability_score, caption_dist = self.smooth_predict_optimized(image)
        
        # Low stability suggests adversarial perturbation
        is_adversarial = stability_score < self.confidence_threshold
        
        return is_adversarial, stability_score, smoothed_caption, caption_dist
    
    def defend_predict(self, image):
        """
        GPU-optimized robust prediction using randomized smoothing
        """
        smoothed_caption, confidence, _ = self.smooth_predict_optimized(image)
        return smoothed_caption, confidence

def evaluate_defense_optimized(defender, dataloader, clip_model, num_images=100):
    """
    GPU-optimized evaluation of the randomized smoothing defense
    """
    results = {
        'clean_accuracy': [],
        'adversarial_detection': [],
        'caption_similarity': [],
        'defense_effectiveness': []
    }
    
    device = defender.device
    processed_images = 0
    
    # Pre-allocate tensors for better memory management
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if processed_images >= num_images:
                break
                
            batch = {k: v.to(device) if k != "caption" else v for k, v in batch.items()}
            images = batch['image']
            true_captions = batch['caption']
            
            # Process batch of images at once when possible
            batch_size = min(len(images), 4, num_images - processed_images)
            
            for j in range(batch_size):
                if processed_images >= num_images:
                    break
                    
                image = images[j]
                true_caption = true_captions[j]
                
                # Test on clean image
                clean_prediction, clean_confidence = defender.defend_predict(image)
                
                # Simulate adversarial attack (add small perturbation)
                adversarial_image = image + torch.randn_like(image, device=device) * 0.1
                adversarial_image = torch.clamp(adversarial_image, -1, 1)
                
                # Test detection capability
                is_adv, detection_conf, adv_prediction, _ = defender.detect_adversarial(adversarial_image)
                
                # Calculate CLIP similarity for evaluation - batch tokenization
                true_texts = clip.tokenize([true_caption, clean_prediction, adv_prediction]).to(device)
                text_features = clip_model.encode_text(true_texts)
                
                true_text_emb = text_features[0:1]
                clean_text_emb = text_features[1:2]
                adv_text_emb = text_features[2:3]
                
                clean_similarity = F.cosine_similarity(true_text_emb, clean_text_emb).item()
                adv_similarity = F.cosine_similarity(true_text_emb, adv_text_emb).item()
                
                results['clean_accuracy'].append(clean_similarity)
                results['adversarial_detection'].append(is_adv)
                results['caption_similarity'].append(clean_similarity)
                results['defense_effectiveness'].append(adv_similarity)
                
                processed_images += 1
                
                if processed_images % 20 == 0:  # Less frequent printing
                    print(f"Processed {processed_images}/{num_images} images...")
    
    return results

def print_evaluation_results(results):
    """Print evaluation results"""
    print("\n=== Randomized Smoothing Defense Evaluation ===")
    print(f"Average Clean Accuracy: {np.mean(results['clean_accuracy']):.4f}")
    print(f"Adversarial Detection Rate: {np.mean(results['adversarial_detection']):.4f}")
    print(f"Average Caption Similarity: {np.mean(results['caption_similarity']):.4f}")
    print(f"Defense Effectiveness: {np.mean(results['defense_effectiveness']):.4f}")
    print(f"Standard Deviation - Clean: {np.std(results['clean_accuracy']):.4f}")
    print(f"Standard Deviation - Defense: {np.std(results['defense_effectiveness']):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPU-Optimized Randomized Smoothing Defense")
    parser.add_argument("--model", type=str, help="Model name", default='vit-gpt2')
    parser.add_argument("--dataset", type=str, help="Dataset name", default='flickr30k')
    parser.add_argument("--noise_std", type=float, help="Noise standard deviation", default=0.5)
    parser.add_argument("--num_samples", type=int, help="Number of samples for smoothing", default=50)
    parser.add_argument("--confidence_threshold", type=float, help="Confidence threshold", default=0.7)
    parser.add_argument("--num_eval_images", type=int, help="Number of images to evaluate", default=100)
    parser.add_argument("--batch_size", type=int, help="Batch size for GPU processing", default=8)
    
    args = parser.parse_args()
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load models and data
    print("Loading models and dataset...")
    clip_model, clip_preprocessor = clip.load("ViT-B/32", device=device)
    image_processor, tokenizer, model, encoder, image_mean, image_std = load_model(model_name=args.model)
    dataloader = load_dataset(args.dataset, image_processor, batch_size=4)
    
    # Initialize GPU-optimized defense
    print("Initializing GPU-Optimized Randomized Smoothing Defense...")
    defender = RandomizedSmoothingDefense(
        model=model,
        model_name=args.model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        noise_std=args.noise_std,
        num_samples=args.num_samples,
        confidence_threshold=args.confidence_threshold,
        batch_size=args.batch_size
    )
    
    # Evaluate defense with GPU optimization
    print("Evaluating defense...")
    results = evaluate_defense_optimized(defender, dataloader, clip_model, args.num_eval_images)
    
    # Print results
    print_evaluation_results(results)
    
    # GPU memory cleanup
    if device == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU Memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")