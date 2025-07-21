import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
import argparse
import clip
from utils import load_model, load_dataset, predict

class RandomizedSmoothingDefense:
    def __init__(self, model, model_name, tokenizer, image_processor, 
                 noise_std=0.5, num_samples=100, confidence_threshold=0.7):
        """
        Randomized Smoothing Defense for Image-to-Text models
        
        Args:
            model: The image-to-text model (ViT-GPT2 or BLIP)
            model_name: Name of the model
            tokenizer: Model tokenizer
            image_processor: Image processor
            noise_std: Standard deviation of Gaussian noise
            num_samples: Number of noisy samples to generate
            confidence_threshold: Threshold for detection
        """
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.noise_std = noise_std
        self.num_samples = num_samples
        self.confidence_threshold = confidence_threshold
        
    def add_gaussian_noise(self, image, std):
        """Add Gaussian noise to image"""
        noise = torch.randn_like(image) * std
        noisy_image = image + noise
        return torch.clamp(noisy_image, -1, 1)  # Clamp to valid range
    
    def smooth_predict(self, image):
        """
        Perform randomized smoothing prediction
        
        Args:
            image: Input image tensor
            
        Returns:
            smoothed_caption: Most frequent caption
            confidence: Confidence score (0-1)
            caption_distribution: Distribution of captions
        """
        captions = []
        
        # Generate multiple noisy versions and predict
        for _ in range(self.num_samples):
            noisy_image = self.add_gaussian_noise(image, self.noise_std)
            caption = predict(self.model_name, self.model, self.tokenizer, 
                            self.image_processor, noisy_image.unsqueeze(0))
            if isinstance(caption, list):
                caption = caption[0]
            captions.append(caption)
        
        # Count caption frequencies
        caption_counter = Counter(captions)
        most_common_caption, most_common_count = caption_counter.most_common(1)[0]
        
        # Calculate confidence as frequency of most common caption
        confidence = most_common_count / self.num_samples
        
        return most_common_caption, confidence, caption_counter
    
    def detect_adversarial(self, image, original_caption=None):
        """
        Detect if image is adversarial based on caption stability
        
        Args:
            image: Input image tensor
            original_caption: Ground truth caption (optional)
            
        Returns:
            is_adversarial: Boolean indicating if image is adversarial
            confidence: Confidence in the decision
            smoothed_caption: Caption from smoothing
        """
        smoothed_caption, stability_score, caption_dist = self.smooth_predict(image)
        
        # Low stability suggests adversarial perturbation
        is_adversarial = stability_score < self.confidence_threshold
        
        return is_adversarial, stability_score, smoothed_caption, caption_dist
    
    def defend_predict(self, image):
        """
        Robust prediction using randomized smoothing
        
        Args:
            image: Input image tensor
            
        Returns:
            defended_caption: Robust caption prediction
            confidence: Confidence in prediction
        """
        smoothed_caption, confidence, _ = self.smooth_predict(image)
        return smoothed_caption, confidence

def evaluate_defense(defender, dataloader, clip_model, num_images=100):
    """
    Evaluate the randomized smoothing defense
    
    Args:
        defender: RandomizedSmoothingDefense instance
        dataloader: Test dataloader
        clip_model: CLIP model for evaluation
        num_images: Number of images to evaluate
    """
    results = {
        'clean_accuracy': [],
        'adversarial_detection': [],
        'caption_similarity': [],
        'defense_effectiveness': []
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    processed_images = 0
    for i, batch in enumerate(dataloader):
        if processed_images >= num_images:
            break
            
        batch = {k: v.cuda() if k != "caption" else v for k, v in batch.items()}
        images = batch['image']
        true_captions = batch['caption']
        
        for j in range(min(len(images), 2)):  # Process 2 images per batch
            if processed_images >= num_images:
                break
                
            image = images[j]
            true_caption = true_captions[j]
            
            # Test on clean image
            clean_prediction, clean_confidence = defender.defend_predict(image)
            
            # Simulate adversarial attack (add small perturbation)
            adversarial_image = image + torch.randn_like(image) * 0.1
            adversarial_image = torch.clamp(adversarial_image, -1, 1)
            
            # Test detection capability
            is_adv, detection_conf, adv_prediction, _ = defender.detect_adversarial(adversarial_image)
            
            # Calculate CLIP similarity for evaluation
            with torch.no_grad():
                true_text_emb = clip_model.encode_text(clip.tokenize([true_caption]).to(device))
                clean_text_emb = clip_model.encode_text(clip.tokenize([clean_prediction]).to(device))
                adv_text_emb = clip_model.encode_text(clip.tokenize([adv_prediction]).to(device))
                
                clean_similarity = F.cosine_similarity(true_text_emb, clean_text_emb).item()
                adv_similarity = F.cosine_similarity(true_text_emb, adv_text_emb).item()
            
            results['clean_accuracy'].append(clean_similarity)
            results['adversarial_detection'].append(is_adv)
            results['caption_similarity'].append(clean_similarity)
            results['defense_effectiveness'].append(adv_similarity)
            
            processed_images += 1
            
            if processed_images % 10 == 0:
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
    parser = argparse.ArgumentParser(description="Randomized Smoothing Defense")
    parser.add_argument("--model", type=str, help="Model name", default='vit-gpt2')
    parser.add_argument("--dataset", type=str, help="Dataset name", default='flickr30k')
    parser.add_argument("--noise_std", type=float, help="Noise standard deviation", default=0.5)
    parser.add_argument("--num_samples", type=int, help="Number of samples for smoothing", default=50)
    parser.add_argument("--confidence_threshold", type=float, help="Confidence threshold", default=0.7)
    parser.add_argument("--num_eval_images", type=int, help="Number of images to evaluate", default=100)
    
    args = parser.parse_args()
    
    # Load models and data
    print("Loading models and dataset...")
    clip_model, clip_preprocessor = clip.load("ViT-B/32", device='cuda')
    image_processor, tokenizer, model, encoder, image_mean, image_std = load_model(model_name=args.model)
    dataloader = load_dataset(args.dataset, image_processor, batch_size=4)
    
    # Initialize defense
    print("Initializing Randomized Smoothing Defense...")
    defender = RandomizedSmoothingDefense(
        model=model,
        model_name=args.model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        noise_std=args.noise_std,
        num_samples=args.num_samples,
        confidence_threshold=args.confidence_threshold
    )
    
    # Evaluate defense
    print("Evaluating defense...")
    results = evaluate_defense(defender, dataloader, clip_model, args.num_eval_images)
    
    # Print results
    print_evaluation_results(results)
    
    # Demo on single image
    print("\n=== Demo on single image ===")
    for i, batch in enumerate(dataloader):
        batch = {k: v.cuda() if k != "caption" else v for k, v in batch.items()}
        demo_image = batch['image'][0]
        demo_caption = batch['caption'][0]
        
        print(f"True caption: {demo_caption}")
        
        # Original prediction
        original_pred = predict(args.model, model, tokenizer, image_processor, demo_image.unsqueeze(0))
        if isinstance(original_pred, list):
            original_pred = original_pred[0]
        print(f"Original prediction: {original_pred}")
        
        # Defended prediction
        defended_pred, confidence = defender.defend_predict(demo_image)
        print(f"Defended prediction: {defended_pred}")
        print(f"Confidence: {confidence:.4f}")
        
        # Detection test
        is_adv, detection_conf, _, caption_dist = defender.detect_adversarial(demo_image)
        print(f"Detected as adversarial: {is_adv} (confidence: {detection_conf:.4f})")
        print(f"Caption distribution: {dict(list(caption_dist.most_common(3)))}")
        
        break