import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2

class FeatureExtractor:
    def __init__(self):
        # CPU üzerinde hızlı çalışması için MobileNetV2 kullanıyoruz (İnsan ayırt etmede oldukça etkilidir)
        self.device = torch.device("cpu")
        
        # Önceden ImageNet üzerinde eğitilmiş modeli yükle
        model = models.mobilenet_v2(pretrained=True)
        
        # Sınıflandırma katmanını sil, sadece "özellik çıkaran" (feature extractor) katmanları bırak
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Görüntüyü PyTorch'un beklediği formata (tensor) çevirme kuralları
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 128)), # Standart yaya / insan boyutu
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, cv2_image):
        if cv2_image is None or cv2_image.size == 0:
            return None
            
        # OpenCV'nin BGR formatını RGB'ye çevir
        img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        input_tensor = self.preprocess(pil_img)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(input_batch)
            
        # 3 Boyutlu feature map'i 1 Boyutlu vektöre (Embedding) çevir
        features = features.mean([2, 3]) 
        
        # L2 Normalizasyonu (Kosinüs benzerliği hesaplamak için gereklidir)
        features = features / features.norm(p=2, dim=1, keepdim=True)
        
        return features.cpu().numpy().flatten()
