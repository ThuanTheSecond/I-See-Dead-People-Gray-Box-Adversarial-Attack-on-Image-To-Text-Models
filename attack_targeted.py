import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms.functional as F2

import argparse

import clip

from utils import predict
from utils import load_model
from utils import load_dataset
from utils import save_img_and_text

device = 'cuda' if torch.cuda.is_available else 'cpu'

def uap_sgd(model, model_name, encoder, tokenizer, image_processor, image_mean, image_std, clip_model, loader, nb_epoch, eps, c = 0.1, targeted=False, lr=0.01, nb_imgs=1000):
    '''
    INPUT
    model       model
    loader      dataloader
    nb_epoch    number of optimization epochs
    eps         maximum perturbation value (L-infinity) norm
    
    OUTPUT
    delta.data  adversarial perturbation
    losses      losses per iteration
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
        print(f'Cos sim: {cos_sim.mean().item():.4f}')
        print(f'Pred: {model_orig_pred}')
        print(f'Orig: {y[1]}')
        print(f'Target: {y[0]}')
        
        cur_losses = []
        
        for epoch in range(nb_epoch):
            
            # Zero out the gradients
            optimizer.zero_grad()
            # Add the noise to the input
            # x_adv = torch.clamp((x[1] + noise).cuda(), -1, 1)
            x_adv = (x[1] + noise).cuda()
            # Embed the perturbed image
            x_adv_emb = encoder(x_adv)[1]
            # L2 distance
            l2_dist = torch.norm((noise).view(len(noise), -1), p=2, dim=1)

            if not targeted:
                untargeted_loss = nn.CosineSimilarity()(x_adv_emb, x_emb).mean()
                loss = untargeted_loss + c * l2_dist
            else:
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
            
        adv_pred = predict(model_name, model, tokenizer, image_processor, x_adv)
        print(f'After attack:\n\t{adv_pred}')
        
        adv_tokenized = clip.tokenize(adv_pred).cuda()
        with torch.no_grad():
            y_adv_emb = clip_model.encode_text(adv_tokenized) 
            x_adv_emb = clip_model.encode_image(F2.resize(x_adv, (224, 224), antialias=True))
        
        clip_score_after = F.cosine_similarity(x_adv_emb, y_adv_emb).mean()
        
        save_img_and_text(F2.resize(x_adv[0], (224, 224)), adv_pred, image_mean, image_std, eps, i, target_img=False, targeted=True, adv=True)
        total_losses.append(cur_losses)
        clip_losses.append((clip_score_before, clip_score_after))
        #print(f'Total current losses: {cur_losses}')
        print(f'CLIP loss before and after: {clip_score_before, clip_score_after}')
        
        torch.cuda.empty_cache()
        
    return total_losses, clip_losses

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="Image-to-Text Attack")
    parser.add_argument("--model", type=str, help="Model name (str)", default='blip')
    parser.add_argument("--dataset", type=str, help="Dataset name (str)", default='flickr30k')
    parser.add_argument("--eps", type=float, help="Epsilon value (float)", default=50/255)
    parser.add_argument("--n_epochs", type=int, help="Number of epochs (int)", default=1000)
    parser.add_argument("--n_imgs", type=int, help="Number of images (int)", default=1000)
    args = parser.parse_args()
    
    model_name  = args.model
    dataset     = args.dataset
    eps         = args.eps
    nb_epoch    = args.n_epochs
    n_imgs      = args.n_imgs
    
    print(f"Starting targeted attack with:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset}")
    print(f"  Epsilon: {eps}")
    print(f"  Epochs: {nb_epoch}")
    print(f"  Images: {n_imgs}")

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
                                        c=0.1,
                                        targeted=True,
                                        lr=0.01,
                                        nb_imgs=n_imgs)
    
    mean_loss = 0
    for loss in total_losses:
        last_loss = loss[-1]
        mean_loss += last_loss
        
    mean_before_loss, mean_after_loss = 0, 0
    for before_loss, after_loss in clip_losses:
        mean_before_loss += before_loss
        mean_after_loss += after_loss
        
    mean_loss /= len(total_losses)
    print(f'Mean last loss: {mean_loss}')
    mean_before_loss /= len(clip_losses)
    print(f'Mean CLIP loss before: {mean_before_loss}')
    mean_after_loss /= len(clip_losses)
    print(f'Mean CLIP loss after: {mean_after_loss}')
    
    before_variance = sum([((x - mean_before_loss) ** 2) for x, y in clip_losses]) / len(clip_losses)
    before_std = before_variance ** 0.5
    print(f'STD CLIP loss before: {before_std}')
    
    after_variance = sum([((y - mean_after_loss) ** 2) for x, y in clip_losses]) / len(clip_losses)
    after_std = after_variance ** 0.5
    print(f'STD CLIP loss after: {after_std}')