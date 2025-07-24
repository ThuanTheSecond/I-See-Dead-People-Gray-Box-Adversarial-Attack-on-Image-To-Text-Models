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
import scipy.ndimage as ndimage  # Thêm import cho median filter
from utils import predict
class ClipTransformDefender:
    def __init__(self, clip_model, model, model_name, tokenizer, image_processor, 
                 detection_threshold=0.85, transform_var_threshold=0.08, transform_mean_threshold=0.75,
                 device='cuda'):
        self.clip_model = clip_model
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        
        # Định nghĩa các biến đổi đầu vào (đã sửa lỗi median filter)
        self.transformations = [
            ('original', lambda x: x),
            ('jpeg_75', lambda x: self._jpeg_compress(x, 75)),
            ('jpeg_90', lambda x: self._jpeg_compress(x, 90)),
            ('gaussian_blur', transforms.GaussianBlur(kernel_size=3)),
            ('median_blur', self._median_filter),  # Sử dụng hàm custom đã sửa
            ('bit_depth', self._bit_depth_reduction),
            ('color_jitter', transforms.ColorJitter(brightness=0.1, contrast=0.1)),
        ]
        
        self.threshold = detection_threshold
        self.transform_var_threshold = transform_var_threshold
        self.transform_mean_threshold = transform_mean_threshold

    def _median_filter(self, img, kernel_size=3):
        """Hàm median filter custom sử dụng scipy"""
        np_img = img.cpu().numpy()
        if len(np_img.shape) == 3:
            np_img = np_img.transpose(1, 2, 0)
        
        filtered = ndimage.median_filter(np_img, size=kernel_size)
        
        if len(filtered.shape) == 3 and filtered.shape[2] == 3:
            filtered = filtered.transpose(2, 0, 1)
            
        return torch.from_numpy(filtered).to(self.device)

    def _jpeg_compress(self, img, quality=90):
        """Optimized JPEG compression"""
        img_pil = TF.to_pil_image((img + 1)/2)  # Chuyển từ [-1,1] sang [0,1]
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return TF.to_tensor(Image.open(buffer)).to(self.device) * 2 - 1

    def _bit_depth_reduction(self, img, bits=5):
        """Bit depth reduction with range validation"""
        img = torch.clamp(img, -1, 1)
        max_val = 2**bits - 1
        img_scaled = ((img + 1) / 2 * max_val).round() / max_val
        return img_scaled * 2 - 1

    def detect_attack(self, image, caption):
        """Phát hiện tấn công với xử lý lỗi cải tiến"""
        try:
            image_resized = TF.resize(image, (224, 224), antialias=True)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_resized.unsqueeze(0))
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
                            transformed_features = self.clip_model.encode_image(transformed_resized.unsqueeze(0))
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

    def defend(self, image, original_caption=None):
        """Phòng thủ với xử lý lỗi cải tiến"""
        try:
            if original_caption:
                is_adversarial, confidence, attack_type = self.detect_attack(image, original_caption)
            else:
                is_adversarial, confidence, attack_type = True, 0.5, "unknown"
            
            defense_info = {
                'is_adversarial': is_adversarial,
                'detection_confidence': confidence,
                'attack_type': attack_type,
                'transformations': []
            }
            
            if not is_adversarial and original_caption:
                return original_caption, 1.0, defense_info
            
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
                return original_caption if original_caption else "", 0.0, defense_info
            
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
            
            return best_caption, confidence, defense_info
            
        except Exception as e:
            print(f"Defense error: {str(e)[:100]}")
            return original_caption if original_caption else "", 0.0, {'error': str(e)}

    def _apply_transform_and_predict(self, image, name, transform):
        """Áp dụng biến đổi và dự đoán với xử lý lỗi"""
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
                image_features = self.clip_model.encode_image(transformed_resized.unsqueeze(0))
                text_tokens = clip.tokenize([caption]).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                similarity = F.cosine_similarity(image_features, text_features).item()
            
            return name, caption, similarity
            
        except Exception as e:
            print(f"Transform {name} error: {str(e)[:100]}")
            raise e