import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import io
import numpy as np
import clip
from collections import Counter

class ClipTransformDefender:
    def __init__(self, clip_model, model, model_name, tokenizer, image_processor, 
                 detection_threshold=0.75, device='cuda'):
        """
        Phòng thủ kết hợp CLIP và Input Transformation
        
        Args:
            clip_model: Mô hình CLIP đã được huấn luyện
            model: Mô hình image captioning (vit-gpt2)
            model_name: Tên mô hình
            tokenizer: Tokenizer cho mô hình
            image_processor: Image processor cho mô hình
            detection_threshold: Ngưỡng phát hiện tấn công
            device: Thiết bị tính toán
        """
        self.clip_model = clip_model
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.threshold = detection_threshold
        self.device = device
        
        # Định nghĩa các biến đổi đầu vào
        self.transformations = [
            ('original', lambda x: x),
            ('jpeg_75', lambda x: self._jpeg_compress(x, quality=75)),
            ('jpeg_90', lambda x: self._jpeg_compress(x, quality=90)),
            ('gaussian_blur', transforms.GaussianBlur(kernel_size=3)),
            ('median_blur', self._median_filter),
            ('bit_depth', self._bit_depth_reduction),
            ('color_jitter', transforms.ColorJitter(brightness=0.1, contrast=0.1)),
        ]
        
    def _jpeg_compress(self, img, quality=90):
        """Áp dụng JPEG compression"""
        buffer = io.BytesIO()
        img_pil = TF.to_pil_image(img)
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img = TF.to_tensor(Image.open(buffer)).to(self.device)
        # Đảm bảo định dạng chính xác
        if img.shape[0] == 3:
            img = img * 2 - 1  # Chuyển [0,1] sang [-1,1]
        return img
    
    def _median_filter(self, img, kernel_size=3):
        """Áp dụng median filter"""
        # Chuyển sang numpy để dễ lọc
        np_img = img.cpu().numpy().transpose(1, 2, 0)
        from scipy import ndimage
        filtered = np.zeros_like(np_img)
        for c in range(np_img.shape[2]):
            filtered[:, :, c] = ndimage.median_filter(np_img[:, :, c], size=kernel_size)
        tensor_img = torch.from_numpy(filtered.transpose(2, 0, 1)).to(self.device)
        return tensor_img
    
    def _bit_depth_reduction(self, img, bits=5):
        """Giảm bit depth của ảnh"""
        max_val = 2**bits - 1
        img_scaled = ((img + 1) / 2 * max_val).round() / max_val
        return img_scaled * 2 - 1
    
    def detect_attack(self, image, caption):
        """
        Phát hiện tấn công đối nghịch sử dụng CLIP
        
        Args:
            image: Ảnh đầu vào
            caption: Caption được dự đoán
            
        Returns:
            is_adversarial: Boolean cho biết có phải tấn công không
            confidence: Độ tin cậy của phát hiện
            attack_type: Loại tấn công (targeted/untargeted/none)
        """
        # Chuẩn bị ảnh và caption cho CLIP
        image_resized = TF.resize(image, (224, 224), antialias=True)
        
        with torch.no_grad():
            # Tính toán embedding cho ảnh và caption
            image_features = self.clip_model.encode_image(image_resized.unsqueeze(0))
            text_tokens = clip.tokenize([caption]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens)
            
            # Đo lường độ tương đồng ngữ nghĩa
            similarity = F.cosine_similarity(image_features, text_features).item()
            
            # Kiểm tra tính nhất quán khi biến đổi
            transform_similarities = []
            for name, transform in self.transformations[1:]: # bỏ qua 'original'
                transformed = transform(image)
                transformed_resized = TF.resize(transformed, (224, 224), antialias=True)
                transformed_features = self.clip_model.encode_image(transformed_resized.unsqueeze(0))
                transform_sim = F.cosine_similarity(image_features, transformed_features).item()
                transform_similarities.append(transform_sim)
            
            # Tính độ biến thiên khi biến đổi
            transform_variance = np.var(transform_similarities)
            transform_mean = np.mean(transform_similarities)
            
            # Đánh giá dựa trên các metrics
            is_semantic_inconsistent = similarity < self.threshold
            is_unstable_to_transforms = transform_variance > 0.03 or transform_mean < 0.85
            
            # Đánh giá loại tấn công
            if is_semantic_inconsistent and is_unstable_to_transforms:
                is_adversarial = True
                confidence = 0.7 + (1.0 - similarity) * 0.3  # 0.7-1.0
                # Phân loại loại tấn công
                if similarity < 0.6:  # Sự khác biệt rất lớn -> khả năng cao là targeted
                    attack_type = "targeted"
                else:  # Sự khác biệt vừa phải -> có thể là untargeted
                    attack_type = "untargeted"
            elif is_semantic_inconsistent:
                is_adversarial = True
                confidence = 0.6 + (1.0 - similarity) * 0.2  # 0.6-0.8
                attack_type = "unknown"
            else:
                is_adversarial = False
                confidence = 0.0
                attack_type = "none"
                
            return is_adversarial, confidence, attack_type
    
    def defend(self, image, original_caption=None):
        """
        Phòng thủ tấn công đối nghịch bằng Input Transformation
        
        Args:
            image: Ảnh đầu vào (có thể bị tấn công)
            original_caption: Caption ban đầu (nếu có)
            
        Returns:
            best_caption: Caption sau khi phòng thủ
            confidence: Độ tin cậy của caption
            defense_info: Thông tin về phòng thủ
        """
        # Kiểm tra nếu ảnh bị tấn công
        if original_caption:
            is_adversarial, confidence, attack_type = self.detect_attack(image, original_caption)
        else:
            # Nếu không có caption gốc, giả định ảnh bị tấn công
            is_adversarial = True
            confidence = 0.5
            attack_type = "unknown"
        
        defense_info = {
            'is_adversarial': is_adversarial,
            'detection_confidence': confidence,
            'attack_type': attack_type,
            'transformations': []
        }
        
        # Nếu không phải ảnh đối nghịch, trả về caption gốc
        if not is_adversarial and original_caption:
            return original_caption, 1.0, defense_info
        
        # Áp dụng các biến đổi và thu thập caption
        captions = []
        similarities = []
        
        for name, transform in self.transformations:
            try:
                # Áp dụng biến đổi
                transformed = transform(image)
                
                # Dự đoán caption
                from utils import predict
                caption = predict(
                    self.model_name, 
                    self.model, 
                    self.tokenizer, 
                    self.image_processor, 
                    transformed.unsqueeze(0)
                )[0]
                
                # Đánh giá độ tin cậy với CLIP
                transformed_resized = TF.resize(transformed, (224, 224), antialias=True)
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(transformed_resized.unsqueeze(0))
                    text_tokens = clip.tokenize([caption]).to(self.device)
                    text_features = self.clip_model.encode_text(text_tokens)
                    similarity = F.cosine_similarity(image_features, text_features).item()
                
                captions.append(caption)
                similarities.append(similarity)
                
                defense_info['transformations'].append({
                    'name': name,
                    'caption': caption,
                    'similarity': similarity
                })
            except Exception as e:
                print(f"Error with transformation {name}: {e}")
        
        # Chiến lược phòng thủ dựa trên loại tấn công
        if attack_type == "targeted":
            # Tìm caption phổ biến nhất (ensemble voting)
            caption_counter = Counter(captions)
            most_common_caption, count = caption_counter.most_common(1)[0]
            vote_confidence = count / len(captions)
            
            # Nếu không có caption rõ ràng chiếm ưu thế, chọn caption có độ tương đồng cao nhất
            if vote_confidence < 0.4:
                best_idx = np.argmax(similarities)
                best_caption = captions[best_idx]
                confidence = similarities[best_idx]
            else:
                best_caption = most_common_caption
                confidence = vote_confidence
                
        else:  # untargeted hoặc unknown
            # Chọn caption có độ tương đồng CLIP cao nhất
            best_idx = np.argmax(similarities)
            best_caption = captions[best_idx]
            confidence = similarities[best_idx]
        
        defense_info['defense_method'] = 'ensemble_voting' if attack_type == 'targeted' else 'best_similarity'
        defense_info['defense_confidence'] = confidence
        
        return best_caption, confidence, defense_info

def test_defender(image_path, model_name='vit-gpt2'):
    """Test function cho defender"""
    from PIL import Image
    import matplotlib.pyplot as plt
    from utils import load_model
    
    # Load models
    clip_model, _ = clip.load("ViT-B/32", device='cuda')
    image_processor, tokenizer, model, _, _, _ = load_model(model_name)
    
    # Load ảnh test
    image = Image.open(image_path).convert("RGB")
    img_tensor = TF.to_tensor(image).cuda() * 2 - 1  # Chuyển sang [-1,1]
    
    # Khởi tạo defender
    defender = ClipTransformDefender(
        clip_model=clip_model,
        model=model, 
        model_name=model_name,
        tokenizer=tokenizer,
        image_processor=image_processor
    )
    
    # Tạo nhiễu đối nghịch giả lập
    noise = torch.randn_like(img_tensor) * 0.05
    adv_img = torch.clamp(img_tensor + noise, -1, 1)
    
    # Dự đoán caption gốc và caption bị tấn công
    from utils import predict
    orig_caption = predict(model_name, model, tokenizer, image_processor, img_tensor.unsqueeze(0))[0]
    adv_caption = predict(model_name, model, tokenizer, image_processor, adv_img.unsqueeze(0))[0]
    
    # Phát hiện và phòng thủ
    is_adv, conf, attack_type = defender.detect_attack(adv_img, adv_caption)
    defended_caption, def_conf, def_info = defender.defend(adv_img, orig_caption)
    
    # In kết quả
    print(f"Original caption: {orig_caption}")
    print(f"Adversarial caption: {adv_caption}")
    print(f"Is adversarial: {is_adv} (confidence: {conf:.4f}, type: {attack_type})")
    print(f"Defended caption: {defended_caption} (confidence: {def_conf:.4f})")
    print(f"Defense method: {def_info['defense_method']}")
    
    return defender

if __name__ == "__main__":
    # Có thể test với một ảnh mẫu
    # test_defender("path/to/image.jpg")
    
    # Hoặc chạy đầy đủ trên dataset
    import argparse
    from utils import load_model, load_dataset
    import clip
    import pandas as pd
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description="CLIP + Input Transformation Defense")
    parser.add_argument("--model", type=str, default="vit-gpt2", help="Model name")
    parser.add_argument("--dataset", type=str, default="flickr30k", help="Dataset name")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--test_image", type=str, default="", help="Single test image path (optional)")
    args = parser.parse_args()
    
    # Test với một ảnh cụ thể nếu được chỉ định
    if args.test_image:
        print(f"Testing with single image: {args.test_image}")
        defender = test_defender(args.test_image, args.model)
        exit(0)
    
    # Chạy đánh giá trên dataset
    print(f"Running evaluation on {args.dataset} with model {args.model}, {args.num_samples} samples")
    
    # Load models
    print("Loading models...")
    clip_model, _ = clip.load("ViT-B/32", device='cuda')
    image_processor, tokenizer, model, encoder, _, _ = load_model(args.model)
    
    # Load dataset
    print("Loading dataset...")
    dataloader = load_dataset(args.dataset, image_processor, batch_size=1)
    
    # Khởi tạo defender
    print("Initializing defender...")
    defender = ClipTransformDefender(
        clip_model=clip_model,
        model=model, 
        model_name=args.model,
        tokenizer=tokenizer,
        image_processor=image_processor
    )
    
    # Thực hiện đánh giá
    results = []
    print("Starting evaluation...")
    
    for i, batch in tqdm(enumerate(dataloader), total=args.num_samples):
        if i >= args.num_samples:
            break
        
        try:
            # Chuẩn bị dữ liệu
            batch = {k: v.cuda() if k != "caption" else v for k, v in batch.items()}
            image = batch['image'][0]  # Lấy ảnh đầu tiên trong batch
            
            # Tạo nhiễu đối nghịch giả lập (có thể thay thế bằng tấn công thực tế)
            noise = torch.randn_like(image) * 0.05
            adv_img = torch.clamp(image + noise, -1, 1)
            
            # Dự đoán caption gốc và caption bị tấn công
            from utils import predict
            orig_caption = predict(args.model, model, tokenizer, image_processor, image.unsqueeze(0))[0]
            adv_caption = predict(args.model, model, tokenizer, image_processor, adv_img.unsqueeze(0))[0]
            
            # Phát hiện và phòng thủ
            is_adv, conf, attack_type = defender.detect_attack(adv_img, adv_caption)
            defended_caption, def_conf, def_info = defender.defend(adv_img, orig_caption)
            
            # Lưu kết quả
            result = {
                'sample_id': i,
                'original_caption': orig_caption,
                'adversarial_caption': adv_caption,
                'defended_caption': defended_caption,
                'is_adversarial': is_adv,
                'detection_confidence': conf,
                'attack_type': attack_type,
                'defense_confidence': def_conf,
                'defense_method': def_info['defense_method']
            }
            
            results.append(result)
            
            # In kết quả mẫu
            if i < 3 or i % 10 == 0:
                print(f"\n=== Sample {i} ===")
                print(f"Original caption: {orig_caption}")
                print(f"Adversarial caption: {adv_caption}")
                print(f"Is adversarial: {is_adv} (confidence: {conf:.4f}, type: {attack_type})")
                print(f"Defended caption: {defended_caption} (confidence: {def_conf:.4f})")
                print(f"Defense method: {def_info['defense_method']}")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
    
    # Tính toán và in thống kê
    if results:
        # Chuyển kết quả thành DataFrame
        df = pd.DataFrame(results)
        
        # Lưu kết quả
        output_file = f"defense_results_{args.model}_{args.dataset}.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        
        # Tính toán thống kê
        detection_accuracy = df['is_adversarial'].mean()
        defense_confidence_avg = df['defense_confidence'].mean()
        
        # Phân tích phân phối loại tấn công được phát hiện
        attack_type_counts = df['attack_type'].value_counts()
        defense_method_counts = df['defense_method'].value_counts()
        
        # In thống kê
        print("\n=== Evaluation Statistics ===")
        print(f"Total samples processed: {len(df)}")
        print(f"Detection accuracy: {detection_accuracy:.4f}")
        print(f"Average defense confidence: {defense_confidence_avg:.4f}")
        print(f"\nDetected attack types:")
        for attack_type, count in attack_type_counts.items():
            print(f"  - {attack_type}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"\nDefense methods used:")
        for method, count in defense_method_counts.items():
            print(f"  - {method}: {count} ({count/len(df)*100:.1f}%)")
        
        # Vẽ biểu đồ
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        attack_type_counts.plot(kind='bar')
        plt.title('Detected Attack Types')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        defense_method_counts.plot(kind='bar')
        plt.title('Defense Methods Used')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f"defense_results_{args.model}_{args.dataset}.png")
        print(f"Charts saved to defense_results_{args.model}_{args.dataset}.png")