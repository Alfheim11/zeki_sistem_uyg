import os
import numpy as np
import shutil
import threading
import cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from ultralytics import YOLO  # İŞTE YENİ SİLAHIMIZ

# --- DERİN ÖĞRENME ---
# Tensorflow uyarılarını susturalım
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# --- AYARLAR ---
DATABASE_DIR = "vehicle_database"
SIMILARITY_THRESHOLD = 0.25  # Limit
EXACT_MATCH_THRESHOLD = 0.05
N_NEIGHBORS = 5

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Global Değişkenler
feature_extractor = None
knn_clf = None  # DİKKAT: Değişkenin asıl adı bu!
yolo_model = None 
stored_features = []
stored_labels = []

def init_models():
    """Hem ResNet hem YOLO modelini yükler"""
    global feature_extractor, yolo_model
    try:
        # 1. YOLO (Nesne Bulucu)
        print("[SİSTEM] YOLOv8 Modeli yükleniyor...")
        yolo_model = YOLO("yolov8n.pt") 
        
        # 2. ResNet (Özellik Çıkarıcı)
        print("[SİSTEM] ResNet50 Modeli yükleniyor...")
        base = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        feature_extractor = Model(inputs=base.input, outputs=base.output)
        
        print("[SİSTEM] Tüm motorlar hazır!")
        train_database()
    except Exception as e:
        print(f"[HATA] Model yükleme sorunu: {e}")

def detect_and_crop_car(img_path):
    """
    YOLO kullanarak resimdeki arabayı bulur ve etrafını keser (Crop).
    """
    try:
        img_cv = cv2.imread(img_path)
        if img_cv is None: return None

        results = yolo_model(img_cv, verbose=False)
        boxes = results[0].boxes
        
        max_area = 0
        best_crop = img_cv 

        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id in [2, 3, 5, 7]: # Araç sınıfları
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                
                if area > max_area:
                    max_area = area
                    best_crop = img_cv[y1:y2, x1:x2] 
        
        best_crop_rgb = cv2.cvtColor(best_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(best_crop_rgb)
        return pil_img

    except Exception as e:
        print(f"YOLO Hatası ({img_path}): {e}")
        return None 

def extract_features(img_path):
    """Önce arabayı keser, sonra analiz eder."""
    try:
        cropped_img = detect_and_crop_car(img_path)
        if cropped_img is None: 
            cropped_img = image.load_img(img_path, target_size=(224, 224))
        
        cropped_img = cropped_img.resize((224, 224))
        img_array = image.img_to_array(cropped_img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        features = feature_extractor.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Analiz Hatası: {e}")
        return None

def train_database():
    global knn_clf, stored_features, stored_labels
    stored_features = []
    stored_labels = []
    
    print("[SİSTEM] Veritabanı YOLO ile taranıyor (Biraz sürebilir)...")

    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)

    for root, dirs, files in os.walk(DATABASE_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path_parts = root.split(os.sep)
                if len(path_parts) >= 2:
                    label = path_parts[-1] if path_parts[-2] == DATABASE_DIR else f"{path_parts[-2]} {path_parts[-1]}"
                else:
                    label = os.path.basename(root)

                vector = extract_features(os.path.join(root, file))
                if vector is not None:
                    stored_features.append(vector)
                    stored_labels.append(label)

    if len(stored_features) > 0:
        k = min(N_NEIGHBORS, len(stored_features))
        # KNN Modelini burada 'knn_clf' adıyla oluşturuyoruz
        knn_clf = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights='distance')
        knn_clf.fit(stored_features, stored_labels)
        print(f"[SİSTEM] Veritabanı Hazır! {len(stored_features)} araç öğrenildi.")
    else:
        print("[UYARI] Veritabanı boş.")

# --- ARAYÜZ ---
class ProCarAI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("YOLO + ResNet Araç Tanıma")
        self.geometry("1000x800")
        
        self.lbl_head = ctk.CTkLabel(self, text="YOLO DESTEKLİ ARAÇ TANIMA", font=("Arial", 24, "bold"))
        self.lbl_head.pack(pady=20)
        
        self.img_panel = ctk.CTkLabel(self, text="Fotoğraf Bekleniyor...", width=700, height=450, fg_color="#222", corner_radius=10)
        self.img_panel.pack(pady=10)
        
        self.lbl_status = ctk.CTkLabel(self, text="Sistem Yükleniyor...", font=("Arial", 18), text_color="orange")
        self.lbl_status.pack(pady=5)
        
        self.lbl_info = ctk.CTkLabel(self, text="", text_color="#aaa")
        self.lbl_info.pack(pady=5)
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=20)
        
        ctk.CTkButton(btn_frame, text="ANALİZ ET", command=self.analyze, width=200, height=50).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="VERİ EKLE", command=self.add_data, width=200, height=50, fg_color="green").pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="YENİLE", command=self.refresh, width=100, height=50, fg_color="gray").pack(side="left", padx=10)
        
        self.file_path = None
        self.ready = False
        threading.Thread(target=self.start_engine).start()

    def start_engine(self):
        init_models()
        self.ready = True
        self.lbl_status.configure(text="SİSTEM HAZIR", text_color="#2ecc71")

    def refresh(self):
        self.lbl_status.configure(text="Veritabanı Yeniden Taranıyor...", text_color="yellow")
        self.update()
        train_database()
        self.lbl_status.configure(text="SİSTEM GÜNCEL", text_color="#2ecc71")

    def analyze(self):
        if not self.ready: return
        path = filedialog.askopenfilename()
        if not path: return
        self.file_path = path
        
        img = Image.open(path)
        ctk_img = ctk.CTkImage(img, size=(600, 400))
        self.img_panel.configure(image=ctk_img, text="")
        
        self.lbl_status.configure(text="YOLO Nesne Arıyor...", text_color="cyan")
        self.lbl_info.configure(text="")
        self.update()
        
        threading.Thread(target=self.process).start()

    def process(self):
        # DÜZELTME BURADA YAPILDI
        if knn_clf is None: 
            self.lbl_status.configure(text="HATA: Veritabanı boş!", text_color="red")
            return
        
        vector = extract_features(self.file_path)
        if vector is None:
            self.lbl_status.configure(text="HATA: Araba Bulunamadı!", text_color="red")
            return

        # ESKİ: distances, indices = knn_classifier.kneighbors(...) -> YANLIŞ
        # YENİ: knn_clf kullanıyoruz
        distances, indices = knn_clf.kneighbors([vector], n_neighbors=min(N_NEIGHBORS, len(stored_features)))
        
        closest_dist = distances[0][0]
        closest_name = stored_labels[indices[0][0]]
        avg_dist = np.mean(distances[0])
        
        neighbors = [stored_labels[i] for i in indices[0]]
        prediction = max(set(neighbors), key=neighbors.count)
        
        if closest_dist < EXACT_MATCH_THRESHOLD:
            res_text = f"✅ KESİN EŞLEŞME: {closest_name}"
            color = "#2ecc71"
        elif avg_dist > SIMILARITY_THRESHOLD:
            res_text = "❌ TANIMLANAMADI"
            color = "red"
            # Thread içinde messagebox açmak bazen tkinter'ı kilitler ama deneyelim
            # İdeal olan ana thread'e sinyal göndermektir.
        else:
            res_text = f"✅ TAHMİN: {prediction}"
            color = "#3498db"
            
        self.lbl_status.configure(text=res_text, text_color=color)
        self.lbl_info.configure(text=f"Mesafe: {avg_dist:.4f} | Komşular: {neighbors}")

    def add_data(self):
        marka = simpledialog.askstring("Giriş", "Marka:")
        if not marka: return
        model = simpledialog.askstring("Giriş", "Model:")
        if not model: return
        
        files = filedialog.askopenfilenames()
        if not files: return
        
        target = os.path.join(DATABASE_DIR, marka.upper(), model.upper())
        os.makedirs(target, exist_ok=True)
        
        for f in files:
            shutil.copy(f, os.path.join(target, os.path.basename(f)))
            
        self.refresh()
        messagebox.showinfo("OK", "Eklendi!")

if __name__ == "__main__":
    app = ProCarAI()
    app.mainloop()
