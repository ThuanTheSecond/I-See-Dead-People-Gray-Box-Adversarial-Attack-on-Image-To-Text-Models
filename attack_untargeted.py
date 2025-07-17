import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse

import clip

from utils import predict
from utils import load_model
from utils import load_dataset
from utils import save_img_and_text

device = 'cuda' if torch.cuda.is_available else 'cpu'

def uap_sgd(model, tokenizer, image_processor, clip_model, clip_preprocessor, loader, nb_epoch, eps, c = 0.1, targeted=False, lr=0.01, nb_imgs=1000):
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
    encoder = model.encoder.cuda()
    # imgs counter
    imgs_counter = 0
    # caption counter
    ys = []
    for i, batch in enumerate(loader):
        batch = {k: v.cuda() if k!="caption" else v for k, v in batch.items()}
        x = batch['image']
        y = batch['caption']
        
        # Forward pass of the end-to-end model
        model_orig_pred = predict(model, tokenizer, image_processor, x)[0]
        
        if model_orig_pred in ys or len(y[0]) > 77:
            continue
        else:
            ys = []
        # append y to ys to pass similar images
        ys.append(model_orig_pred)
    
        # Tokenize the texts for CLIP
        pred_text = clip.tokenize([model_orig_pred]).cuda()
        true_text = clip.tokenize(y).cuda()
                    
        # Embed both texts
        pred_text_features = clip_model.encode_text(pred_text)
        true_text_features = clip_model.encode_text(true_text)
            
        # Cosine similarity for filtering
        cos_sim = F.cosine_similarity(pred_text_features, true_text_features)

        # If the prediction is not close to the ground truth, continue
        if cos_sim < 0.7:
            continue
        
        y_true_emb = clip_model.encode_text(true_text) 
        x_true_emb = clip_model.encode_image(x)
        
        clip_score_before = F.cosine_similarity(x_true_emb, y_true_emb).item()
        
        imgs_counter += 1
        if imgs_counter == nb_imgs:
            break
        
        # Create a noise tensor
        noise = torch.zeros((1, 3, 224, 224), device=device, requires_grad=True)
        # Define the optimizer
        optimizer = Adam([noise], lr=lr)
        # Define the scheduler
        scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=30, factor=0.1, cooldown=30)
        
        # Save the original image embedding
        x_emb = encoder(x)[1].detach()
            
        save_img_and_text(x, model_orig_pred, eps, i, adv=False)
        print(f'Cos sim: {cos_sim.item():.4f}')
        print(f'Pred: {model_orig_pred}')
        print(f'Orig: {y[0]}')
        
        cur_losses = []
        
        for epoch in range(nb_epoch):
            # Zero out the gradients
            optimizer.zero_grad()
            # Add the noise to the input
            x_adv = torch.clamp((x + noise).cuda(), -1, 1)
            # Embed the perturbed image
            x_adv_emb = encoder(x_adv)[1]
            # L2 distance
            l2_dist = torch.norm((noise).view(len(noise), -1), p=2, dim=1)

            if not targeted:
                untargeted_loss = nn.CosineSimilarity()(x_adv_emb, x_emb).mean()
                loss = untargeted_loss + c * l2_dist
                        
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
            
        adv_pred = predict(model, tokenizer, image_processor, x_adv)[0]
        print(f'After attack:\n\t{adv_pred}')
        
        adv_tokenized = clip.tokenize(adv_pred).cuda()
        y_adv_emb = clip_model.encode_text(adv_tokenized) 
        x_adv_emb = clip_model.encode_image(x_adv)
        
        clip_score_after = F.cosine_similarity(x_adv_emb, y_adv_emb).item()
        
        save_img_and_text(x_adv, adv_pred, eps, i, adv=True)
        total_losses.append(cur_losses)
        clip_losses.append((clip_score_before, clip_score_after))
        #print(f'Total current losses: {cur_losses}')
        print(f'CLIP loss before and after: {clip_score_before, clip_score_after}')
    return total_losses, clip_losses

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description="Image-to-Text Attack")
    parser.add_argument("--model", type=str, help="Model name (str)", default='vit-gpt2')
    parser.add_argument("--dataset", type=str, help="Dataset name (str)", default='flickr30k')
    parser.add_argument("--eps", type=float, help="Epsilon value (float)", default=50/255)
    parser.add_argument("--n_epochs", type=int, help="Number of epochs (int)", default=1000)
    parser.add_argument("--n_imgs", type=int, help="Number of images (int)", default=1000)
    args = parser.parse_args()
    
    model    = args.model
    dataset  = args.dataset
    eps      = args.eps
    nb_epoch = args.n_epochs
    n_imgs   = args.n_imgs

    clip_model, clip_preprocessor = clip.load("ViT-B/32", device='cuda')
    #image_processor, tokenizer, model = load_model(model_name=model)
    image_processor, tokenizer, model, encoder, mean, std = load_model(model_name=model)
    dataloader = load_dataset(dataset, image_processor)
    
    total_losses, clip_losses = uap_sgd(model=model, tokenizer=tokenizer, image_processor=image_processor, clip_model=clip_model, clip_preprocessor=clip_preprocessor, loader=dataloader, nb_epoch=nb_epoch, eps=eps, c = 0.1, targeted=False, lr=0.01, nb_imgs=n_imgs)
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