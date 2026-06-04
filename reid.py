import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# Yüz tanıma opsiyonel - InsightFace kullan (dlib yerine)
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

class FeatureExtractor:
    def __init__(self):
        # MobileNetV3-Small: MobileNetV2'den ~2x daha hızlı, Re-ID için yeterli hassasiyet
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # Sınıflandırma katmanını çıkar, sadece özellik çıkaran kısımları bırak
        # MobileNetV3-Small: features + avgpool (classifier hariç)
        self.model = nn.Sequential(model.features, model.avgpool)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Daha küçük girdi boyutu = daha hızlı inference
        # 128x64 piksel — yaya Re-ID için standart boyuttan küçük ama yeterli
        self.preprocess = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # InsightFace yüz tanıma modeli - lazy loading (ilk frame geldiğinde başlatılacak)
        self.face_app = None
        self.face_app_initialized = False

    def extract(self, cv2_image):
        if cv2_image is None or cv2_image.size == 0:
            return None

        h, w = cv2_image.shape[:2]
        if h < 10 or w < 5:
            return None

        try:
            img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            input_tensor = self.preprocess(pil_img)
            input_batch = input_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(input_batch)

            # 3D feature map → 1D vektör (Embedding)
            features = features.mean([2, 3])

            # L2 Normalizasyonu (Kosinüs benzerliği için)
            features = features / (features.norm(p=2, dim=1, keepdim=True) + 1e-8)

            return features.cpu().numpy().flatten()
        except Exception:
            return None

    def extract_face(self, cv2_image):
        """Yüz tanıma için embedding çıkar - InsightFace ile (lazy loading)"""
        if not INSIGHTFACE_AVAILABLE:
            return None
            
        if cv2_image is None or cv2_image.size == 0:
            return None

        h, w = cv2_image.shape[:2]
        if h < 50 or w < 50:  # Yüz için minimum boyut
            return None

        # Lazy loading - ilk çağrıda InsightFace'i başlat
        if not self.face_app_initialized:
            try:
                self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
                self.face_app.prepare(ctx_id=-1, det_size=(640, 640))
                self.face_app_initialized = True
                print("InsightFace initialized successfully (lazy loading)")
            except Exception as e:
                print(f"InsightFace initialization failed: {e}")
                print("Face recognition will be disabled")
                self.face_app = None
                self.face_app_initialized = True  # Tekrar deneme yapma
                return None

        if self.face_app is None:
            return None

        try:
            # InsightFace ile yüz tespiti ve embedding
            faces = self.face_app.get(cv2_image)
            if len(faces) == 0:
                return None
            
            # İlk yüzün embedding'ini al
            return faces[0].embedding
        except Exception:
            return None
