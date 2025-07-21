import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms.functional as F2

import argparse
from collections import Counter
import clip

from utils import predict
from utils import load_model
from utils import load_dataset
from utils import save_img_and_text

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RandomizedSmoothingDefender:
    def __init__(self, model, model_name, tokenizer, image_processor, 
                 noise_std=0.1, num_samples=30, batch_size=8):
        """
        Lightweight Randomized Smoothing Defense for integration with attack
        """
        self.model = model.eval()
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.noise_std = noise_std
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.device = device
        
    def add_noise_batch(self, image, std, num_samples):
        """Add Gaussian noise to create batch of noisy images"""
        # Create batch of the same image
        if len(image.shape) == 3:  # Single image
            image = image.unsqueeze(0)
        batch_images = image.repeat(num_samples, 1, 1, 1)
        # Generate noise for entire batch
        noise = torch.randn_like(batch_images) * std
        noisy_batch = torch.clamp(batch_images + noise, -1, 1)
        return noisy_batch
    
    def batch_predict(self, image_batch):
        """Batch prediction for multiple noisy images"""
        all_captions = []
        
        for i in range(0, len(image_batch), self.batch_size):
            batch_slice = image_batch[i:i+self.batch_size]
            
            with torch.no_grad():
                captions = predict(self.model_name, self.model, self.tokenizer, 
                                 self.image_processor, batch_slice)
                all_captions.extend(captions)
        
        return all_captions
    
    def defend(self, image):
        """Apply randomized smoothing to defend against adversarial attack"""
        noisy_batch = self.add_noise_batch(image, self.noise_std, self.num_samples)
        captions = self.batch_predict(noisy_batch)
        
        # Find most common caption
        caption_counter = Counter(captions)
        most_common_caption, count = caption_counter.most_common(1)[0]
        confidence = count / self.num_samples
        
        return most_common_caption, confidence, caption_counter

def evaluate_attack_and_defense(original_caption, attacked_caption, defended_caption, 
                               clip_model, clip_original_score, clip_attacked_score):
    """Calculate metrics for attack effectiveness and defense success"""
    # Tokenize for CLIP evaluation
    tokens = clip.tokenize([original_caption, attacked_caption, defended_caption]).to(device)
    
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
    
    # Calculate semantic similarities between captions
    orig_to_attacked = F.cosine_similarity(text_features[0:1], text_features[1:2]).item()
    orig_to_defended = F.cosine_similarity(text_features[0:1], text_features[2:3]).item()
    attacked_to_defended = F.cosine_similarity(text_features[1:2], text_features[2:3]).item()
    
    metrics = {
        'orig_attacked_similarity': orig_to_attacked,
        'orig_defended_similarity': orig_to_defended,
        'attacked_defended_similarity': attacked_to_defended,
        'clip_score_original': clip_original_score,
        'clip_score_attacked': clip_attacked_score,
        'defense_improvement': orig_to_defended - orig_to_attacked
    }
    
    return metrics

def uap_sgd_with_defense(model, model_name, encoder, tokenizer, image_processor, image_mean, image_std, 
                         clip_model, defender, loader, nb_epoch, eps, c=0.1, lr=0.01, nb_imgs=1000):
    '''
    Targeted attack with integrated defense evaluation
    '''
    total_losses = []
    metrics_list = []
    # image encoder
    encoder = encoder.cuda()
    # imgs counter
    imgs_counter = 0
    
    for i, batch in enumerate(loader):
        batch = {k: v.cuda() if k!="caption" else v for k, v in batch.items()}
        x = batch['image']
        y = batch['caption']
        
        # Make sure we have enough images in batch
        if x.shape[0] < 6:
            continue
            
        x = torch.stack([x[0], x[5]])  # taking the first and the 6th since they're different
        y = [y[0], y[5]]  # taking the first and the 6th since they're different
        
        # Forward pass of the end-to-end model
        model_orig_pred = predict(model_name, model, tokenizer, image_processor, x)
    
        # Tokenize the texts for CLIP
        pred_texts = clip.tokenize(model_orig_pred).cuda()
        true_texts = clip.tokenize(y).cuda()
                    
        # Embed both texts
        with torch.no_grad():
            pred_texts_features = clip_model.encode_text(pred_texts)
            true_texts_features = clip_model.encode_text(true_texts)
            
        # Cosine similarity for filtering
        cos_sim = F.cosine_similarity(pred_texts_features, true_texts_features)

        # If the prediction is not close to the ground truth, continue
        if cos_sim[0] < 0.7 or cos_sim[1] < 0.7 or len(y[0]) > 76 or len(y[1]) > 76:
            continue
        
        with torch.no_grad():
            y_true_emb = clip_model.encode_text(true_texts) 
            x_true_emb = clip_model.encode_image(F2.resize(x, (224, 224), antialias=True))
        
        clip_score_before = F.cosine_similarity(x_true_emb, y_true_emb).mean()
        
        imgs_counter += 1
        if imgs_counter == nb_imgs:
            break
        
        # Create a noise tensor
        noise = torch.zeros_like(x[0].unsqueeze(0), device=device, requires_grad=True)
        # Define the optimizer
        optimizer = Adam([noise], lr=lr)
        # Define the scheduler
        scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=30, factor=0.1, cooldown=30)
        
        # Save the original target image embedding
        with torch.no_grad():
            x_emb = encoder(x)[1][0].detach()
            
        save_img_and_text(F2.resize(x[0], (224, 224)), model_orig_pred[0], image_mean, image_std, eps, i, target_img=True, targeted=True, adv=False)
        save_img_and_text(F2.resize(x[1], (224, 224)), model_orig_pred[1], image_mean, image_std, eps, i, target_img=False, targeted=True, adv=False)
        print(f'\n==== Image Pair {imgs_counter} ====')
        print(f'Cos sim: {cos_sim.mean().item():.4f}')
        print(f'Original caption (source image): {model_orig_pred[1]}')
        print(f'Target caption: {model_orig_pred[0]}')
        
        cur_losses = []
        
        for epoch in range(nb_epoch):
            # Zero out the gradients
            optimizer.zero_grad()
            # Add the noise to the input
            x_adv = (x[1] + noise).cuda()
            # Embed the perturbed image
            x_adv_emb = encoder(x_adv)[1]
            # L2 distance
            l2_dist = torch.norm((noise).view(len(noise), -1), p=2, dim=1)

            # Targeted attack loss (make output similar to target)
            targeted_loss = 1 - nn.CosineSimilarity()(x_adv_emb, x_emb).mean()
            loss = targeted_loss + c * l2_dist
                        
            # Backpropagate the gradients
            loss.backward()
            # Step according to gradient
            optimizer.step()
            # Scheduler step
            scheduler.step(loss)
            # Project to epsilon ball
            noise.data.clamp_(-eps, eps)
            # Save loss
            cur_losses.append(loss.data.item())
            
            if epoch % 100 == 99:
                print(f'Epoch #{epoch+1} loss: {loss.data[0]:.4f}')
            
        # Get the adversarial prediction
        adv_pred = predict(model_name, model, tokenizer, image_processor, x_adv)
        print(f'\nAfter attack:')
        print(f'Adversarial caption: {adv_pred[0]}')
        
        # Apply defense
        print("\nApplying randomized smoothing defense...")
        defended_caption, confidence, caption_dist = defender.defend(x_adv[0])
        print(f'Defended caption: {defended_caption}')
        print(f'Defense confidence: {confidence:.4f}')
        print(f'Top 3 captions from defense: {dict(caption_dist.most_common(3))}')
        
        # Tokenize for CLIP evaluation
        adv_tokenized = clip.tokenize(adv_pred).cuda()
        with torch.no_grad():
            y_adv_emb = clip_model.encode_text(adv_tokenized) 
            x_adv_emb = clip_model.encode_image(F2.resize(x_adv, (224, 224), antialias=True))
        
        clip_score_after = F.cosine_similarity(x_adv_emb, y_adv_emb).mean()
        
        # Calculate metrics for both attack and defense
        metrics = evaluate_attack_and_defense(
            original_caption=model_orig_pred[1],
            attacked_caption=adv_pred[0],
            defended_caption=defended_caption,
            clip_model=clip_model,
            clip_original_score=clip_score_before.item(),
            clip_attacked_score=clip_score_after.item()
        )
        
        # Print metrics
        print("\n=== Attack & Defense Metrics ===")
        print(f"Original to Attacked Caption Similarity: {metrics['orig_attacked_similarity']:.4f}")
        print(f"Original to Defended Caption Similarity: {metrics['orig_defended_similarity']:.4f}")
        print(f"Attacked to Defended Caption Similarity: {metrics['attacked_defended_similarity']:.4f}")
        print(f"CLIP Score Before Attack: {metrics['clip_score_original']:.4f}")
        print(f"CLIP Score After Attack: {metrics['clip_score_attacked']:.4f}")
        print(f"Defense Improvement: {metrics['defense_improvement']:.4f}")
        
        # Save adversarial image with both captions
        save_img_and_text(
            F2.resize(x_adv[0], (224, 224)), 
            f"Attack: {adv_pred[0]}\nDefense: {defended_caption}", 
            image_mean, image_std, eps, i, 
            target_img=False, targeted=True, adv=True
        )
        
        total_losses.append(cur_losses)
        metrics_list.append(metrics)
        
        torch.cuda.empty_cache()
        
    return total_losses, metrics_list

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="Image-to-Text Attack with Defense")
    parser.add_argument("--model", type=str, help="Model name (str)", default='vit-gpt2')
    parser.add_argument("--dataset", type=str, help="Dataset name (str)", default='flickr30k')
    parser.add_argument("--eps", type=float, help="Epsilon value (float)", default=50/255)
    parser.add_argument("--n_epochs", type=int, help="Number of epochs (int)", default=1000)
    parser.add_argument("--n_imgs", type=int, help="Number of images (int)", default=1000)
    parser.add_argument("--noise_std", type=float, help="Defense noise standard deviation", default=0.1)
    parser.add_argument("--num_samples", type=int, help="Defense number of samples", default=30)
    args = parser.parse_args()
    
    model_name  = args.model
    dataset     = args.dataset
    eps         = args.eps
    nb_epoch    = args.n_epochs
    n_imgs      = args.n_imgs
    noise_std   = args.noise_std
    num_samples = args.num_samples

    print(f"Loading models and dataset...")
    clip_model, clip_preprocessor = clip.load("ViT-B/32", device=device)
    image_processor, tokenizer, model, encoder, image_mean, image_std = load_model(model_name=model_name)
    dataloader = load_dataset(dataset, image_processor, batch_size=6)
    
    # Initialize defender
    print(f"Initializing randomized smoothing defender...")
    defender = RandomizedSmoothingDefender(
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        image_processor=image_processor,
        noise_std=noise_std,
        num_samples=num_samples
    )
    
    print(f"Starting attack with integrated defense evaluation...")
    total_losses, metrics_list = uap_sgd_with_defense(
        model=model,
        model_name=model_name,
        encoder=encoder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_mean=image_mean,
        image_std=image_std,
        clip_model=clip_model,
        defender=defender,
        loader=dataloader,
        nb_epoch=nb_epoch,
        eps=eps,
        c=0.1,
        lr=0.01,
        nb_imgs=n_imgs
    )
    
    # Calculate average metrics
    avg_metrics = {k: 0.0 for k in metrics_list[0].keys()}
    for metrics in metrics_list:
        for k, v in metrics.items():
            avg_metrics[k] += v
    
    for k in avg_metrics.keys():
        avg_metrics[k] /= len(metrics_list)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Number of images processed: {len(metrics_list)}")
    print(f"Average attack success (1-similarity): {1-avg_metrics['orig_attacked_similarity']:.4f}")
    print(f"Average defense effectiveness: {avg_metrics['orig_defended_similarity']:.4f}")
    print(f"Average defense improvement: {avg_metrics['defense_improvement']:.4f}")
    
    # Calculate standard deviations
    std_metrics = {k: 0.0 for k in metrics_list[0].keys()}
    for metrics in metrics_list:
        for k, v in metrics.items():
            std_metrics[k] += (v - avg_metrics[k])**2
    
    for k in std_metrics.keys():
        std_metrics[k] = (std_metrics[k] / len(metrics_list))**0.5
    
    print("\n=== Standard Deviations ===")
    print(f"Attack Success StdDev: {std_metrics['orig_attacked_similarity']:.4f}")
    print(f"Defense Effectiveness StdDev: {std_metrics['orig_defended_similarity']:.4f}")
    print(f"Defense Improvement StdDev: {std_metrics['defense_improvement']:.4f}")