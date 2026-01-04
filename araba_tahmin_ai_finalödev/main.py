import os
import numpy as np
import shutil
import threading
import cv2
import pickle  
import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from ultralytics import YOLO

# --- DERİN ÖĞRENME MOTORU ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# --- AYARLAR ---
DATABASE_DIR = "vehicle_database"
CACHE_FILE = "beyin_verisi.pkl" # Öğrenilen bilgileri buraya kaydedeceğiz

# Hassasiyet Ayarları
SIMILARITY_THRESHOLD = 0.18  
EXACT_MATCH_THRESHOLD = 0.05
N_NEIGHBORS = 5

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Global Değişkenler
feature_extractor = None
knn_clf = None  
yolo_model = None 
stored_features = []
stored_labels = []

def init_models():
    """Modelleri yükler ve veritabanını (varsa dosyadan) çeker."""
    global feature_extractor, yolo_model
    try:
        # 1. YOLO (Göz)
        print("[SİSTEM] YOLOv8 Modeli yükleniyor...")
        yolo_model = YOLO("yolov8n.pt") 
        
        # 2. ResNet (Beyin)
        print("[SİSTEM] ResNet50 Modeli yükleniyor...")
        base = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        feature_extractor = Model(inputs=base.input, outputs=base.output)
        
        # 3. Veritabanı Kontrolü
        # İlk açılışta dosyadan okumayı dene, yoksa tara.
        train_database(force_scan=False)
        
    except Exception as e:
        print(f"[HATA] Model yükleme sorunu: {e}")

def enhance_image(img_cv):
    """Görüntü İyileştirme (Hızlı Mod)"""
    try:
        if img_cv is None: return None
        
        # A) UPSCALE (Kaliteli Büyütme)
        height, width = img_cv.shape[:2]
        target_width = 800  
        
        if width < target_width:
            scale = target_width / width
            img_cv = cv2.resize(img_cv, (int(width*scale), int(height*scale)), interpolation=cv2.INTER_LANCZOS4)

        # B) SHARPEN (Keskinleştirme)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img_cv = cv2.filter2D(img_cv, -1, kernel)

        # C) CLAHE (Işık Dengele)
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return final

    except Exception as e:
        print(f"Görüntü işleme hatası: {e}")
        return img_cv 

def detect_and_crop_car(img_path):
    """YOLO ile arabayı bulur, keser."""
    try:
        img_cv = cv2.imread(img_path)
        if img_cv is None: return None

        img_cv = enhance_image(img_cv)

        results = yolo_model(img_cv, verbose=False)
        boxes = results[0].boxes
        
        max_area = 0
        best_crop = img_cv 

        found = False
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id in [2, 3, 5, 7]: 
                found = True
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
    """Resmi vektöre çevirir."""
    try:
        cropped_img = detect_and_crop_car(img_path)
        if cropped_img is None: return None
        
        cropped_img = cropped_img.resize((224, 224))
        img_array = image.img_to_array(cropped_img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        features = feature_extractor.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Analiz Hatası: {e}")
        return None

def train_database(force_scan=False):
    """
    Veritabanını eğitir VEYA kayıtlı dosyadan yükler.
    force_scan=True ise zorla yeniden tarar (Yeni veri eklenince).
    """
    global knn_clf, stored_features, stored_labels
    
    # 1. DURUM: Dosya varsa ve zorla tarama istenmiyorsa -> YÜKLE
    if not force_scan and os.path.exists(CACHE_FILE):
        print("[SİSTEM] Kayıtlı beyin verisi bulundu! Hızlı yükleniyor...")
        try:
            with open(CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
                stored_features = data['features']
                stored_labels = data['labels']
                knn_clf = data['model']
            print(f"[SİSTEM] Hazır! {len(stored_features)} araç hafızaya alındı (Önbellekten).")
            return
        except Exception as e:
            print(f"[UYARI] Kayıt dosyası bozuk, yeniden taranacak: {e}")

    # 2. DURUM: Dosya yoksa veya 'Yenile' dendi ise -> TARA VE KAYDET
    print("[SİSTEM] Veritabanı taranıyor... (Bu işlem biraz sürebilir)")
    stored_features = []
    stored_labels = []

    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)

    for root, dirs, files in os.walk(DATABASE_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                path_parts = root.split(os.sep)
                if len(path_parts) >= 2:
                    if path_parts[-1] == DATABASE_DIR: label = "Bilinmeyen"
                    elif path_parts[-2] == DATABASE_DIR: label = path_parts[-1]
                    else: label = f"{path_parts[-2]} {path_parts[-1]}"
                else:
                    label = os.path.basename(root)

                full_path = os.path.join(root, file)
                vector = extract_features(full_path)
                
                if vector is not None:
                    stored_features.append(vector)
                    stored_labels.append(label)

    if len(stored_features) > 0:
        k = min(N_NEIGHBORS, len(stored_features))
        knn_clf = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights='distance')
        knn_clf.fit(stored_features, stored_labels)
        
        # --- KAYDETME İŞLEMİ (PICKLE) ---
        print("[SİSTEM] Bilgiler diske kaydediliyor (Sonraki açılış hızlanacak)...")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump({
                'features': stored_features,
                'labels': stored_labels,
                'model': knn_clf
            }, f)
        
        print(f"[SİSTEM] Veritabanı Hazır! {len(stored_features)} araç öğrenildi.")
    else:
        print("[UYARI] Veritabanı boş!")

# --- ARAYÜZ (GUI) ---
class ProCarAI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PRO CAR AI - Turbo Edition")
        self.geometry("1100x850")
        
        self.lbl_head = ctk.CTkLabel(self, text="YOLO + RESNET ARAÇ TANIMA (TURBO)", font=("Roboto", 26, "bold"))
        self.lbl_head.pack(pady=20)
        
        self.img_panel = ctk.CTkLabel(self, text="Sistem Başlatılıyor...", width=800, height=500, fg_color="#1a1a1a", corner_radius=15)
        self.img_panel.pack(pady=10)
        
        self.lbl_status = ctk.CTkLabel(self, text="Modeller Yükleniyor...", font=("Roboto", 20), text_color="orange")
        self.lbl_status.pack(pady=10)
        
        self.lbl_info = ctk.CTkLabel(self, text="", text_color="#aaa", font=("Consolas", 12))
        self.lbl_info.pack(pady=5)
        
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=20)
        
        ctk.CTkButton(btn_frame, text="ANALİZ ET", command=self.analyze, width=200, height=50, font=("Roboto", 14, "bold")).pack(side="left", padx=15)
        ctk.CTkButton(btn_frame, text="VERİ EKLE", command=self.add_data, width=200, height=50, fg_color="#27ae60", hover_color="#2ecc71", font=("Roboto", 14, "bold")).pack(side="left", padx=15)
        ctk.CTkButton(btn_frame, text="YENİLE", command=self.refresh, width=120, height=50, fg_color="#7f8c8d", hover_color="#95a5a6").pack(side="left", padx=15)
        
        self.file_path = None
        self.ready = False
        
        threading.Thread(target=self.start_engine, daemon=True).start()

    def start_engine(self):
        init_models()
        self.ready = True
        self.lbl_status.configure(text="SİSTEM HAZIR", text_color="#2ecc71")
        self.img_panel.configure(text="Fotoğraf Bekleniyor...")

    def refresh(self):
        # Kullanıcı 'Yenile'ye basarsa zorla yeniden tara (force_scan=True)
        self.lbl_status.configure(text="Veritabanı Yeniden Taranıyor...", text_color="yellow")
        self.update()
        threading.Thread(target=self._refresh_thread, daemon=True).start()

    def _refresh_thread(self):
        train_database(force_scan=True) # <-- Zorla tara ve kaydet
        self.lbl_status.configure(text="SİSTEM GÜNCELLENDİ", text_color="#2ecc71")

    def analyze(self):
        if not self.ready: 
            messagebox.showwarning("Bekle", "Sistem henüz yüklenmedi!")
            return
        path = filedialog.askopenfilename()
        if not path: return
        self.file_path = path
        
        try:
            img = Image.open(path)
            ratio = min(800/img.width, 500/img.height)
            new_size = (int(img.width*ratio), int(img.height*ratio))
            ctk_img = ctk.CTkImage(img, size=new_size)
            
            self.img_panel.configure(image=ctk_img, text="")
            self.lbl_status.configure(text="Analiz Ediliyor...", text_color="cyan")
            self.lbl_info.configure(text="")
            self.update()
            
            threading.Thread(target=self.process, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Hata", f"Resim açılamadı: {e}")

    def process(self):
        if knn_clf is None: 
            self.lbl_status.configure(text="HATA: Veritabanı boş!", text_color="red")
            return
        
        vector = extract_features(self.file_path)
        if vector is None:
            self.lbl_status.configure(text="HATA: Araç Tespit Edilemedi!", text_color="red")
            return

        distances, indices = knn_clf.kneighbors([vector], n_neighbors=min(N_NEIGHBORS, len(stored_features)))
        closest_dist = distances[0][0]
        closest_name = stored_labels[indices[0][0]]
        neighbors = [stored_labels[i] for i in indices[0]]
        prediction = max(set(neighbors), key=neighbors.count)
        
        print("\n" + "="*40)
        print(f"ANALİZ: {os.path.basename(self.file_path)}")
        print(f"Tahmin: {prediction} | Mesafe: {closest_dist:.4f}")
        print("="*40 + "\n")

        if closest_dist < EXACT_MATCH_THRESHOLD:
            res_text = f"✅ KESİN EŞLEŞME: {closest_name}"
            color = "#2ecc71"
        elif closest_dist > SIMILARITY_THRESHOLD:
            res_text = f"❌ TANIMLANAMADI (Mesafe: {closest_dist:.2f})"
            color = "#e74c3c"
        elif neighbors.count(prediction) < 3:
             res_text = f"⚠️ KARARSIZIM: {prediction} olabilir..."
             color = "#f39c12"
        else:
            res_text = f"✅ TAHMİN: {prediction}"
            color = "#3498db"
            
        self.lbl_status.configure(text=res_text, text_color=color)
        conf = max(0, (1 - closest_dist) * 100) 
        self.lbl_info.configure(text=f"Güven: %{conf:.1f} | Mesafe: {closest_dist:.4f}")

    def add_data(self):
        marka = simpledialog.askstring("Giriş", "Marka:")
        if not marka: return
        model = simpledialog.askstring("Giriş", "Model:")
        if not model: return
        
        files = filedialog.askopenfilenames()
        if not files: return
        
        target_dir = os.path.join(DATABASE_DIR, marka.upper().strip(), model.upper().strip())
        os.makedirs(target_dir, exist_ok=True)
        
        for f in files:
            shutil.copy(f, os.path.join(target_dir, os.path.basename(f)))
            
        messagebox.showinfo("Tamam", "Veriler eklendi! Veritabanı güncelleniyor...")
        self.refresh() # Otomatik yenile ve kaydet

if __name__ == "__main__":
    app = ProCarAI()
    app.mainloop()
