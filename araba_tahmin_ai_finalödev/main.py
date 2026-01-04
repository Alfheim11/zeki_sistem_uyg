import os
import numpy as np
import shutil
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

# --- DERİN ÖĞRENME KÜTÜPHANELERİ ---
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# --- SİSTEM AYARLARI ---
DATABASE_DIR = "vehicle_database"

# BURASI ÇOK KRİTİK:
# 0.05 -> Birebir aynısıysa kabul et.
# 0.23 -> Benziyorsa kabul et (Bunu 0.27'den düşürdüm ki Skoda'ya Mercedes demesin).
# Eğer tanıdığı arabaları reddederse bunu 0.24 veya 0.25 yap.
EXACT_MATCH_THRESHOLD = 0.05 
SIMILARITY_THRESHOLD = 0.23   
N_NEIGHBORS = 5

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Global Değişkenler
feature_extractor_model = None
knn_classifier = None
stored_features = []
stored_labels = []

def load_extraction_model():
    """Derin öğrenme modelini yükler."""
    global feature_extractor_model
    try:
        # ResNet50'yi sadece özellik çıkarıcı olarak kullanıyoruz (include_top=False)
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        feature_extractor_model = Model(inputs=base_model.input, outputs=base_model.output)
        print("[SİSTEM] ResNet50 Modeli Yüklendi.")
        train_database_model()
    except Exception as e:
        print(f"[HATA] Model yüklenirken hata: {e}")

def extract_features(img_path):
    """Görüntüden sayısal öznitelik vektörü çıkarır."""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = feature_extractor_model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"[HATA] Görüntü işlenemedi: {img_path} - {e}")
        return None

def train_database_model():
    """Veritabanındaki görüntüleri tarar ve KNN algoritmasını eğitir."""
    global knn_classifier, stored_features, stored_labels
    
    stored_features = []
    stored_labels = []
    
    print("[SİSTEM] Veritabanı indeksleniyor...")

    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)

    # Klasör yapısını tara (Marka/Model)
    for root, dirs, files in os.walk(DATABASE_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Etiket oluşturma (Marka + Model)
                path_parts = root.split(os.sep)
                if len(path_parts) >= 2:
                    if path_parts[-2] == DATABASE_DIR:
                         label = path_parts[-1]
                    else:
                         label = f"{path_parts[-2]} {path_parts[-1]}"
                else:
                    label = os.path.basename(root)

                full_path = os.path.join(root, file)
                vector = extract_features(full_path)
                
                if vector is not None:
                    stored_features.append(vector)
                    stored_labels.append(label)

    data_count = len(stored_features)
    
    # KNN Eğitimi
    k_value = min(N_NEIGHBORS, data_count)
    
    if data_count > 0:
        # Distance ağırlıklı KNN: Yakındaki komşunun oyu daha değerlidir.
        knn_classifier = KNeighborsClassifier(n_neighbors=k_value, metric='cosine', weights='distance')
        knn_classifier.fit(stored_features, stored_labels)
        print(f"[SİSTEM] Hazır. Toplam {data_count} araç fotoğrafı öğrenildi.")
    else:
        print("[UYARI] Veritabanı boş.")
        knn_classifier = None

# --- ARAYÜZ ---
class CarRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Araç Tanıma Sistemi v2.0")
        self.geometry("1000x750")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Başlık
        self.lbl_title = ctk.CTkLabel(self, text="DERİN ÖĞRENME ARAÇ ANALİZİ", font=("Roboto", 24, "bold"))
        self.lbl_title.grid(row=0, column=0, pady=20)

        # Resim Alanı
        self.img_display = ctk.CTkLabel(self, text="Analiz için fotoğraf seçiniz...", width=650, height=400, corner_radius=10, fg_color="#222")
        self.img_display.grid(row=1, column=0, pady=10)

        # Kontrol Paneli
        self.frame_controls = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_controls.grid(row=2, column=0, pady=20, sticky="ew")

        # Sonuçlar
        self.lbl_result = ctk.CTkLabel(self.frame_controls, text="Sistem Başlatılıyor...", font=("Roboto", 20, "bold"), text_color="orange")
        self.lbl_result.pack(pady=5)
        
        self.lbl_details = ctk.CTkLabel(self.frame_controls, text="", font=("Consolas", 14), text_color="#aaa")
        self.lbl_details.pack(pady=5)

        # Butonlar
        self.frame_buttons = ctk.CTkFrame(self.frame_controls, fg_color="transparent")
        self.frame_buttons.pack(pady=10)

        self.btn_analyze = ctk.CTkButton(self.frame_buttons, text="FOTOĞRAF ANALİZ ET", command=self.select_image, 
                                         height=50, width=200, font=("Roboto", 14, "bold"), fg_color="#106EBE")
        self.btn_analyze.pack(side="left", padx=10)

        # İŞTE GERİ GELEN BUTON:
        self.btn_add = ctk.CTkButton(self.frame_buttons, text="VERİ EKLE (LİSTE)", command=self.add_batch_data, 
                                     height=50, width=200, font=("Roboto", 14, "bold"), fg_color="#27ae60")
        self.btn_add.pack(side="left", padx=10)
        
        self.btn_refresh = ctk.CTkButton(self.frame_buttons, text="YENİLE", command=self.refresh_db, 
                                         height=50, width=100, fg_color="#7f8c8d")
        self.btn_refresh.pack(side="left", padx=10)

        self.selected_img_path = None
        self.is_ready = False
        threading.Thread(target=self.init_system).start()

    def init_system(self):
        load_extraction_model()
        self.is_ready = True
        self.lbl_result.configure(text="SİSTEM HAZIR", text_color="#2ecc71")

    def refresh_db(self):
        self.lbl_result.configure(text="Veritabanı Güncelleniyor...", text_color="yellow")
        self.update()
        train_database_model()
        self.lbl_result.configure(text="SİSTEM GÜNCEL", text_color="#2ecc71")

    def select_image(self):
        if not self.is_ready: return
        file_path = filedialog.askopenfilename(filetypes=[("Resimler", "*.jpg *.jpeg *.png")])
        if not file_path: return

        self.selected_img_path = file_path
        
        try:
            pil_img = Image.open(file_path)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(600, 400))
            self.img_display.configure(image=ctk_img, text="")
        except: pass
        
        self.lbl_result.configure(text="Analiz Ediliyor...", text_color="yellow")
        self.lbl_details.configure(text="")
        self.update()
        self.after(100, self.predict_vehicle)

    def predict_vehicle(self):
        if knn_classifier is None:
            self.lbl_result.configure(text="Veritabanı Boş!", text_color="red")
            return

        query_vector = extract_features(self.selected_img_path)
        if query_vector is None: return

        # KNN Sorgusu
        distances, indices = knn_classifier.kneighbors([query_vector], n_neighbors=min(N_NEIGHBORS, len(stored_features)))
        
        closest_distance = distances[0][0]
        closest_label = stored_labels[indices[0][0]]
        avg_distance = np.mean(distances[0])

        neighbors = [stored_labels[i] for i in indices[0]]
        predicted_label = max(set(neighbors), key=neighbors.count)
        vote_count = neighbors.count(predicted_label)
        
        debug_msg = f"Mesafe: {avg_distance:.4f} (Limit: {SIMILARITY_THRESHOLD}) | Komşular: {neighbors}"

        # --- FİNAL MANTIK KURGUSU ---

        # 1. KAPI: Birebir Eşleşme (Exact Match)
        # Eğer çok yakınsa (%95+ benzerlik), oylamayı boşver, bu O'dur.
        if closest_distance < EXACT_MATCH_THRESHOLD:
            self.lbl_result.configure(text=f"✅ TESPİT EDİLDİ: {closest_label}", text_color="#2ecc71")
            self.lbl_details.configure(text=f"BİREBİR EŞLEŞME (Veritabanında Kayıtlı)\nMesafe: {closest_distance:.5f}")
            return

        # 2. KAPI: Mesafe Kontrolü (Sallamayı Önleme)
        # Eğer ortalama mesafe limitin üstündeyse, reddet.
        if avg_distance > SIMILARITY_THRESHOLD:
            self.lbl_result.configure(text="❌ TANIMLANAMADI", text_color="red")
            self.lbl_details.configure(text=f"Benzetemedim. Çok farklı görünüyor.\n{debug_msg}")
            
            # Tanınmadıysa ekleme önerisi
            if messagebox.askyesno("Tanınamadı", "Bu aracı tanımıyorum. Veritabanına eklemek ister misin?"):
                self.add_batch_data()
            return

        # 3. KAPI: Jüri Kontrolü (Kararlılık)
        # Eğer mesafe uygunsa ama jüri kararsızsa uyar.
        if vote_count < (N_NEIGHBORS / 2):
            self.lbl_result.configure(text=f"❓ EMİN DEĞİLİM ({predicted_label}?)", text_color="orange")
            self.lbl_details.configure(text=f"Kararsız kaldım.\n{debug_msg}")
        else:
            self.lbl_result.configure(text=f"✅ TAHMİN: {predicted_label}", text_color="#3498db")
            self.lbl_details.configure(text=f"Güvenilir Eşleşme.\n{debug_msg}")

    def add_batch_data(self):
        """Toplu veri ekleme fonksiyonu (Liste şeklinde)"""
        brand = simpledialog.askstring("Veri Girişi", "Marka Adı (Örn: Mercedes):")
        if not brand: return
        model = simpledialog.askstring("Veri Girişi", "Model Adı (Örn: CLS200):")
        if not model: return

        # Klasör isimlerini düzelt
        brand = brand.strip().replace(" ", "_").upper()
        model = model.strip().replace(" ", "_").upper()

        # ÇOKLU SEÇİM (Listeli)
        file_paths = filedialog.askopenfilenames(title="Fotoğrafları Seç (Çoklu Seçim)", 
                                                 filetypes=[("Resimler", "*.jpg *.jpeg *.png")])
        if not file_paths: return

        target_dir = os.path.join(DATABASE_DIR, brand, model)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        count = 0
        for src in file_paths:
            try:
                filename = os.path.basename(src)
                dst = os.path.join(target_dir, filename)
                shutil.copy(src, dst)
                count += 1
            except Exception as e:
                print(f"Kopyalama hatası: {e}")

        # Ekleme bitince veritabanını yenile
        self.refresh_db()
        messagebox.showinfo("Başarılı", f"{count} adet fotoğraf eklendi:\n{brand} -> {model}")

if __name__ == "__main__":
    app = CarRecognitionApp()
    app.mainloop()
