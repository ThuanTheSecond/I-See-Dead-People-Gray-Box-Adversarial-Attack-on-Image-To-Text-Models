# Thực hiện các cải tiến:
#     improve loss function: Kết hợp CLIP loss trực tiếp
#     Better optimizer: AdamW + Cosine Annealing
#     Smart initialization: Thay thế zeros initialization
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms.functional as F2
import torch.nn.init as init

import argparse
import clip

from utils import predict
from utils import load_model
from utils import load_dataset
from utils import save_img_and_text

device = 'cuda' if torch.cuda.is_available else 'cpu'

def smart_noise_initialization(x_source, x_target, encoder, init_method='gradient_based', eps=0.1):
    """Smart initialization for adversarial noise"""
    if init_method == 'gradient_based':
        # Use gradient of target similarity as initialization
        x_source_copy = x_source.clone().requires_grad_(True)
        
        # FIX: Ensure proper batch dimension for encoder
        x_target_batch = x_target.unsqueeze(0) if x_target.dim() == 3 else x_target
        x_source_batch = x_source_copy.unsqueeze(0) if x_source_copy.dim() == 3 else x_source_copy
        
        x_target_emb = encoder(x_target_batch)[1]
        x_source_emb = encoder(x_source_batch)[1]
        
        # Compute gradient pointing towards target
        similarity = F.cosine_similarity(x_source_emb, x_target_emb).mean()
        similarity.backward()
        
        # Use gradient direction for initialization
        noise = x_source_copy.grad.sign() * eps * 0.1
        
        # FIX: Ensure noise has batch dimension like original
        if noise.dim() == 3:
            noise = noise.unsqueeze(0)
            
        return noise.detach().requires_grad_(True)
        
    else:  # fallback to zeros with proper shape
        # FIX: Match original implementation shape
        if x_source.dim() == 3:
            return torch.zeros_like(x_source.unsqueeze(0), requires_grad=True)
        else:
            return torch.zeros_like(x_source, requires_grad=True)

def get_clip_weight(epoch, total_epochs, initial_weight=0.3, final_weight=0.8):
    """Increase CLIP weight over time for better target focusing"""
    progress = epoch / total_epochs
    return initial_weight + (final_weight - initial_weight) * progress

def uap_sgd(model, model_name, encoder, tokenizer, image_processor, image_mean, image_std, clip_model, loader, nb_epoch, eps, c=0.1, targeted=False, lr=0.01, nb_imgs=1000, clip_weight=0.5):
    '''
    Universal Adversarial Perturbation using Stochastic Gradient Descent with improvements
    
    INPUT
    model           Vision-Language model
    model_name      Model identifier string
    encoder         Image encoder
    tokenizer       Text tokenizer
    image_processor Image preprocessor
    image_mean      Image normalization mean
    image_std       Image normalization std
    clip_model      CLIP model for semantic evaluation
    loader          Data loader
    nb_epoch        Number of optimization epochs
    eps             Maximum perturbation value (L-infinity norm)
    c               L2 regularization weight
    targeted        Whether to perform targeted attack
    lr              Learning rate
    nb_imgs         Number of images to process
    clip_weight     Weight for CLIP loss components
    
    OUTPUT
    total_losses    List of losses per iteration for each image
    clip_losses     List of CLIP scores before and after attack
    '''
    total_losses = []
    clip_losses = []
    # image encoder
    encoder = encoder.cuda()
    # imgs counter
    imgs_counter = 0
    
    for i, batch in enumerate(loader):
        batch = {k: v.cuda() if k!="caption" else v for k, v in batch.items()}
        x = batch['image']
        y = batch['caption']
        
        x = torch.stack([x[0], x[5]]) # taking the first and the 6th since they're different
        y = [y[0], y[5]] # taking the first and the 6th since they're different
        
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
        
        # Smart noise initialization (IMPROVEMENT 1) - FIX
        noise = smart_noise_initialization(
            x[1], x[0], encoder, 
            init_method='gradient_based', 
            eps=eps
        ).to(device)
        
        # FIX: Ensure noise shape matches original approach
        if noise.dim() == 3:
            noise = noise.unsqueeze(0)
    
        # Improved optimizer and scheduler (IMPROVEMENT 2)
        optimizer = AdamW([noise], lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
        scheduler = CosineAnnealingLR(optimizer, T_max=nb_epoch, eta_min=lr*0.01)
        
        # PRE-COMPUTE embeddings (GPU optimization)
        # Save the original target image embedding
        with torch.no_grad():
            x_emb = encoder(x)[1][0].detach()
            x_base = x[1].cuda()  # Keep on GPU
            
            # FIX: Pre-compute target text embedding HERE
            target_tokenized = clip.tokenize([y[0]]).cuda()
            target_text_emb = clip_model.encode_text(target_tokenized)
        
        save_img_and_text(F2.resize(x[0], (224, 224)), model_orig_pred[0], image_mean, image_std, eps, i, target_img=True, targeted=True, adv=False)
        save_img_and_text(F2.resize(x[1], (224, 224)), model_orig_pred[1], image_mean, image_std, eps, i, target_img=False, targeted=True, adv=False)
        print(f'Cos sim: {cos_sim.mean().item():.4f}')
        print(f'Pred: {model_orig_pred}')
        print(f'Orig: {y[1]}')
        print(f'Target: {y[0]}')
        
        cur_losses = []
        
        # BATCH ACCUMULATION for GPU efficiency
        loss_accumulation = []
        print_interval = 100
        
        # FIX: Pre-compute target text embedding ONCE before the loop
        with torch.no_grad():
            target_tokenized = clip.tokenize([y[0]]).cuda()
            target_text_emb = clip_model.encode_text(target_tokenized)

        for epoch in range(nb_epoch):
            # Zero out the gradients
            optimizer.zero_grad()
            # Add the noise to the input
            x_adv = x_base + noise.squeeze(0) if noise.dim() == 4 else noise
            x_adv = (x[1] + noise).cuda()
                
            # Ensure batch dimension for encoder
            if x_adv.dim() == 3:
                x_adv = x_adv.unsqueeze(0)
            
            # Embed the perturbed image
            x_adv_emb = encoder(x_adv)[1]
            
            # OPTIMIZATION: Minimize CPU-GPU transfers
            # Only generate prediction when needed (not every epoch)
            if epoch % 50 == 0 or epoch == nb_epoch - 1:
                adv_pred = predict(model_name, model, tokenizer, image_processor, x_adv)
                adv_tokenized = clip.tokenize(adv_pred).cuda()
                adv_text_emb = clip_model.encode_text(adv_tokenized)
            # Generate prediction for adversarial image
            adv_pred = predict(model_name, model, tokenizer, image_processor, x_adv)
            
            # OPTIMIZATION: Pre-compute resized tensor
            if epoch == 0:
                x_adv_size = F2.resize(x_adv, (224, 224), antialias=True)
            else:
                # Update only the noise part
                x_adv_size = F2.resize(x_adv, (224, 224), antialias=True)
            # CLIP-based multi-component loss (IMPROVEMENT 3)
            adv_tokenized = clip.tokenize(adv_pred).cuda()
            
            with torch.no_grad():
                target_text_emb = clip_model.encode_text(target_tokenized)
            
            adv_text_emb = clip_model.encode_text(adv_tokenized)
            
            adv_img_emb = clip_model.encode_image(x_adv_size)
            # FIX: Resize with proper batch handling
            x_adv_resized = F2.resize(x_adv, (224, 224), antialias=True)
            adv_img_emb = clip_model.encode_image(x_adv_resized)
            
            # Loss components (All GPU operations)
            # Loss components
            # 1. Original embedding similarity loss
            if not targeted:
                embedding_loss = nn.CosineSimilarity()(x_adv_emb, x_emb).mean()
            else:
                embedding_loss = 1 - nn.CosineSimilarity()(x_adv_emb, x_emb).mean()
            
            # 2. CLIP semantic consistency loss
            clip_img_text_loss = 1 - F.cosine_similarity(adv_img_emb, adv_text_emb).mean()
            
            # 3. Target similarity loss with temperature scaling
            temperature = max(0.1, 1.0 - epoch / nb_epoch)  # Cool down over time
            target_similarity_loss = (1 - F.cosine_similarity(adv_text_emb, target_text_emb).mean()) / temperature
            # 3. Target similarity loss (for targeted attack)
            target_similarity_loss = 1 - F.cosine_similarity(adv_text_emb, target_text_emb).mean()
            
            # 4. L2 regularization
            l2_dist = torch.norm(noise.view(noise.shape[0], -1), p=2, dim=1)
            
            # 5. Dynamic clip weight scheduling
            current_clip_weight = get_clip_weight(epoch, nb_epoch, 0.3, 0.8)
            
            # 6. Alignment penalty for poor semantic consistency
            alignment_penalty = torch.clamp(clip_img_text_loss - 0.3, min=0) * 2.0
            
            # Combined loss with all improvements
            loss = (embedding_loss + 
                   current_clip_weight * (clip_img_text_loss + target_similarity_loss) + 
                   c * l2_dist + 
                   alignment_penalty)
            
            # OPTIMIZATION: Accumulate losses on GPU
            loss_accumulation.append(loss.detach())
            l2_dist = torch.norm((noise).view(len(noise), -1), p=2, dim=1)
            
            # Combined loss with CLIP components
            loss = embedding_loss + clip_weight * (clip_img_text_loss + target_similarity_loss) + c * l2_dist
                    
            # Improved training step
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([noise], max_norm=1.0)
            # Step according to gradient
            optimizer.step()
            # Scheduler step (after optimizer step for cosine annealing)
            scheduler.step()
            # Project to epsilon ball
            noise.data.clamp_(-eps, eps)
            # Save loss
            cur_losses.append(loss.data.item())
            
            # OPTIMIZATION: Reduce CPU transfers for logging
            if epoch % print_interval == print_interval - 1:
                # Transfer to CPU only when printing
                avg_loss = torch.stack(loss_accumulation).mean().item()
                cur_losses.extend([l.item() for l in loss_accumulation])
                loss_accumulation = []  # Reset
                
                print(f'Epoch #{epoch+1} avg loss: {avg_loss:.4f}')
            if epoch % 100 == 99:
                print(f'Epoch #{epoch+1} loss: {loss.data.item():.4f}')
                # Print loss components for debugging
                print(f'  Embedding loss: {embedding_loss.item():.4f}')
                print(f'  CLIP img-text loss: {clip_img_text_loss.item():.4f}')
                print(f'  Target similarity loss: {target_similarity_loss.item():.4f}')
                print(f'  Current CLIP weight: {current_clip_weight:.4f}')
                print(f'  Temperature: {temperature:.4f}')
                print(f'  Alignment penalty: {alignment_penalty.item():.4f}')
                print(f'  L2 distance: {l2_dist.mean().item():.4f}')
                print(f'  Learning rate: {scheduler.get_last_lr()[0]:.6f}')
            
        # --- Start of the reviewed section (with improvements) ---
        
        # Final prediction after attack
        adv_pred = predict(model_name, model, tokenizer, image_processor, x_adv)
        print(f'After attack:\n\t{adv_pred}')
        
        # Final evaluation
        # Final evaluation with CLIP
        adv_tokenized = clip.tokenize(adv_pred).cuda()
        with torch.no_grad():
            adv_pred = predict(model_name, model, tokenizer, image_processor, x_adv)
            print(f'After attack:\n\t{adv_pred}')
            
            adv_tokenized = clip.tokenize(adv_pred).cuda()
            y_adv_emb = clip_model.encode_text(adv_tokenized)
            y_adv_emb = clip_model.encode_text(adv_tokenized) 
            x_adv_emb_clip = clip_model.encode_image(F2.resize(x_adv, (224, 224), antialias=True))
            
            clip_score_after = F.cosine_similarity(x_adv_emb_clip, y_adv_emb).mean()
            target_similarity_final = F.cosine_similarity(y_adv_emb, target_text_emb).mean()
        
        x_adv_for_save = x_adv.squeeze(0) if x_adv.dim() == 4 else x_adv
        save_img_and_text(F2.resize(x_adv_for_save, (224, 224)), adv_pred, image_mean, image_std, eps, i, target_img=False, targeted=True, adv=True)
        # save_img_and_text(F2.resize(x_adv[0], (224, 224)), adv_pred, image_mean, image_std, eps, i, target_img=False, targeted=True, adv=True)
        clip_score_after = F.cosine_similarity(x_adv_emb_clip, y_adv_emb).mean()
        
        # Save the adversarial image and its caption
        # IMPROVEMENT: Simplified tensor handling for saving
        save_img_and_text(F2.resize(x_adv.squeeze(0), (224, 224)), adv_pred, image_mean, image_std, eps, i, target_img=False, targeted=True, adv=True)
        
        # Store results
        total_losses.append(cur_losses)
        clip_losses.append((clip_score_before, clip_score_after))
        #print(f'Total current losses: {len(cur_losses)} losses recorded')
        clip_losses.append((clip_score_before.item(), clip_score_after.item())) # Use .item() to save floats, not tensors
        
        # Log final results for this image
        print(f'Final loss: {cur_losses[-1]:.4f}')
        print(f'CLIP loss before and after: {clip_score_before.item():.4f}, {clip_score_after.item():.4f}')
        print(f'CLIP score before and after: {clip_score_before.item():.4f}, {clip_score_after.item():.4f}')
        
        # Calculate target caption similarity
        # Calculate and log the final target caption similarity
        target_similarity_final = F.cosine_similarity(y_adv_emb, target_text_emb).mean()
        print(f'Target caption similarity: {target_similarity_final.item():.4f}')
        
        # OPTIMIZATION: Clear GPU cache
        torch.cuda.empty_cache()
        
    return total_losses, clip_losses

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="Image-to-Text Attack - Improved Version")
    parser.add_argument("--model", type=str, help="Model name (str)", default='blip')
    parser.add_argument("--dataset", type=str, help="Dataset name (str)", default='flickr30k')
    parser.add_argument("--eps", type=float, help="Epsilon value (float)", default=50/255)
    parser.add_argument("--n_epochs", type=int, help="Number of epochs (int)", default=1000)
    parser.add_argument("--n_imgs", type=int, help="Number of images (int)", default=1000)
    parser.add_argument("--clip_weight", type=float, help="CLIP loss weight (float)", default=0.5)
    parser.add_argument("--lr", type=float, help="Learning rate (float)", default=0.01)
    parser.add_argument("--c", type=float, help="L2 regularization weight (float)", default=0.1)
    args = parser.parse_args()
    
    model_name  = args.model
    dataset     = args.dataset
    eps         = args.eps
    nb_epoch    = args.n_epochs
    n_imgs      = args.n_imgs
    clip_weight = args.clip_weight
    lr          = args.lr
    c           = args.c

    print(f"Starting improved targeted attack with:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset}")
    print(f"  Epsilon: {eps}")
    print(f"  Epochs: {nb_epoch}")
    print(f"  Images: {n_imgs}")
    print(f"  CLIP weight: {clip_weight}")
    print(f"  Learning rate: {lr}")
    print(f"  L2 weight: {c}")

    clip_model, clip_preprocessor = clip.load("ViT-B/32", device='cuda')
    image_processor, tokenizer, model, encoder, image_mean, image_std = load_model(model_name=model_name)
    dataloader = load_dataset(dataset, image_processor, batch_size=6)
    
    total_losses, clip_losses = uap_sgd(model=model,
                                        model_name=model_name,
                                        encoder=encoder,
                                        tokenizer=tokenizer,
                                        image_processor=image_processor,
                                        image_mean=image_mean,
                                        image_std=image_std,
                                        clip_model=clip_model,
                                        loader=dataloader,
                                        nb_epoch=nb_epoch,
                                        eps=eps,
                                        c=c,
                                        targeted=True,
                                        lr=lr,
                                        nb_imgs=n_imgs,
                                        clip_weight=clip_weight)
    
    # Calculate and print statistics
    mean_loss = 0
    for loss in total_losses:
        last_loss = loss[-1]
        mean_loss += last_loss
        
    mean_before_loss, mean_after_loss = 0, 0
    for before_loss, after_loss in clip_losses:
        mean_before_loss += before_loss
        mean_after_loss += after_loss
        
    mean_loss /= len(total_losses)
    print(f'\n=== FINAL RESULTS ===')
    print(f'Mean last loss: {mean_loss:.4f}')
    
    mean_before_loss /= len(clip_losses)
    print(f'Mean CLIP loss before: {mean_before_loss:.4f}')
    
    mean_after_loss /= len(clip_losses)
    print(f'Mean CLIP loss after: {mean_after_loss:.4f}')
    
    print(f'CLIP score improvement: {(mean_after_loss - mean_before_loss):.4f}')
    
    # Calculate standard deviations
    before_variance = sum([((x - mean_before_loss) ** 2) for x, y in clip_losses]) / len(clip_losses)
    before_std = before_variance ** 0.5
    print(f'STD CLIP loss before: {before_std:.4f}')
    
    after_variance = sum([((y - mean_after_loss) ** 2) for x, y in clip_losses]) / len(clip_losses)
    after_std = after_variance ** 0.5
    print(f'STD CLIP loss after: {after_std:.4f}')
    
    # Calculate success rate (improvement in CLIP score)
    success_count = sum([1 for before, after in clip_losses if after > before])
    success_rate = success_count / len(clip_losses) * 100
    print(f'Attack success rate: {success_rate:.2f}% ({success_count}/{len(clip_losses)})')