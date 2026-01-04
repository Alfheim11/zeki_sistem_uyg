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

# --- SİSTEM KONFİGÜRASYONU ---
DATABASE_DIR = "vehicle_database"  # Veritabanı klasör adı
SIMILARITY_THRESHOLD = 0.27        # Benzerlik eşik değeri (0.27 ideal bir dengedir)
N_NEIGHBORS = 5                    # K-NN algoritması için komşu sayısı

# Arayüz Teması
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Global Değişkenler
feature_extractor_model = None  # Özellik çıkarıcı model
knn_classifier = None           # Sınıflandırma algoritması
stored_features = []            # Kayıtlı vektörler
stored_labels = []              # Kayıtlı etiketler

def load_extraction_model():
    """
    ResNet50 derin öğrenme modelini 'imagenet' ağırlıklarıyla yükler.
    Son katman (classification layer) çıkarılarak sadece özellik çıkarıcı olarak kullanılır.
    """
    global feature_extractor_model
    try:
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        feature_extractor_model = Model(inputs=base_model.input, outputs=base_model.output)
        print("[SİSTEM] Model başarıyla yüklendi: ResNet50")
        train_database_model()
    except Exception as e:
        print(f"[HATA] Model yüklenirken sorun oluştu: {e}")

def extract_features(img_path):
    """
    Verilen görüntü dosyasını okur, ön işleme tabi tutar ve
    model üzerinden geçirerek 2048 boyutlu öznitelik vektörünü çıkarır.
    """
    try:
        # Görüntüyü modelin giriş boyutuna (224x224) yeniden boyutlandır
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # ResNet50 için gerekli ön işlem (normalizasyon)
        img_array = preprocess_input(img_array)
        
        # Özellik çıkarımı
        features = feature_extractor_model.predict(img_array, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"[HATA] Görüntü işlenemedi ({img_path}): {e}")
        return None

def train_database_model():
    """
    Veritabanı klasöründeki tüm görüntüleri tarar, özellik vektörlerini çıkarır
    ve K-Nearest Neighbors (KNN) algoritmasını eğitir.
    """
    global knn_classifier, stored_features, stored_labels
    
    stored_features = []
    stored_labels = []
    
    print("[SİSTEM] Veritabanı taranıyor ve indeksleniyor...")

    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)

    # Klasör yapısını (Marka/Model) tarama
    for root, dirs, files in os.walk(DATABASE_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Etiketleme mantığı: Klasör adlarını birleştirir (Örn: BMW M3)
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
    # Veri sayısı N_NEIGHBORS'dan az ise, mevcut veri sayısı kadar komşuya bakılır.
    k_value = min(N_NEIGHBORS, data_count)
    
    if data_count > 0:
        # Metric: Cosine Similarity (Görüntü benzerliği için Öklid'den daha etkilidir)
        knn_classifier = KNeighborsClassifier(n_neighbors=k_value, metric='cosine', weights='distance')
        knn_classifier.fit(stored_features, stored_labels)
        print(f"[SİSTEM] Eğitim tamamlandı. Toplam Görüntü: {data_count}, Sınıflandırıcı: KNN (K={k_value})")
    else:
        print("[UYARI] Veritabanı boş. Lütfen sisteme görüntü ekleyin.")
        knn_classifier = None

# --- GRAFİKSEL KULLANICI ARAYÜZÜ (GUI) ---
class CarRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Araç Tanıma ve Sınıflandırma Sistemi v1.0")
        self.geometry("1024x768")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Başlık Paneli
        self.lbl_title = ctk.CTkLabel(self, text="DERİN ÖĞRENME TABANLI ARAÇ TANIMA", font=("Roboto", 24, "bold"))
        self.lbl_title.grid(row=0, column=0, pady=20)

        # Görüntüleme Paneli
        self.img_display = ctk.CTkLabel(self, text="Analiz için bir görüntü seçiniz...", 
                                        width=700, height=450, corner_radius=10, fg_color="#2b2b2b")
        self.img_display.grid(row=1, column=0, pady=10)

        # Kontrol ve Bilgi Paneli
        self.frame_controls = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_controls.grid(row=2, column=0, pady=20, sticky="ew")

        # Sonuç Metinleri
        self.lbl_result = ctk.CTkLabel(self.frame_controls, text="Sistem Başlatılıyor...", font=("Roboto", 20, "bold"), text_color="#F39C12")
        self.lbl_result.pack(pady=5)
        
        self.lbl_details = ctk.CTkLabel(self.frame_controls, text="", font=("Consolas", 14), text_color="#BDC3C7")
        self.lbl_details.pack(pady=5)

        # Buton Grubu
        self.frame_buttons = ctk.CTkFrame(self.frame_controls, fg_color="transparent")
        self.frame_buttons.pack(pady=15)

        self.btn_analyze = ctk.CTkButton(self.frame_buttons, text="GÖRÜNTÜ ANALİZİ", command=self.select_image, 
                                         height=45, width=180, font=("Roboto", 14, "bold"), fg_color="#2980B9")
        self.btn_analyze.pack(side="left", padx=10)
        
        self.btn_add_data = ctk.CTkButton(self.frame_buttons, text="VERİ EKLE (EĞİTİM)", command=self.add_batch_data, 
                                          height=45, width=180, font=("Roboto", 14, "bold"), fg_color="#27AE60")
        self.btn_add_data.pack(side="left", padx=10)
        
        self.btn_retrain = ctk.CTkButton(self.frame_buttons, text="MODELİ GÜNCELLE", command=self.refresh_database, 
                                         height=45, width=180, fg_color="#7F8C8D")
        self.btn_retrain.pack(side="left", padx=10)

        self.selected_img_path = None
        self.is_ready = False
        
        # Model yüklemesini arka planda başlat (Arayüz donmasını engeller)
        threading.Thread(target=self.init_system).start()

    def init_system(self):
        load_extraction_model()
        self.is_ready = True
        self.lbl_result.configure(text="SİSTEM HAZIR", text_color="#2ECC71")

    def refresh_database(self):
        self.lbl_result.configure(text="Veritabanı İndeksleniyor...", text_color="#F1C40F")
        self.update()
        train_database_model()
        self.lbl_result.configure(text="VERİTABANI GÜNCELLENDİ", text_color="#2ECC71")

    def display_image(self, path):
        try:
            pil_img = Image.open(path)
            # Görüntüyü arayüze sığacak şekilde yeniden boyutlandır (Aspect Ratio korunarak)
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(600, 400))
            self.img_display.configure(image=ctk_img, text="")
        except Exception as e:
            print(f"Görüntü gösterme hatası: {e}")

    def select_image(self):
        if not self.is_ready:
            messagebox.showwarning("Sistem Uyarısı", "Model henüz yüklenmedi, lütfen bekleyiniz.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Görüntü Dosyaları", "*.jpg *.jpeg *.png")])
        if not file_path: return

        self.selected_img_path = file_path
        self.display_image(file_path)
        
        self.lbl_result.configure(text="Öznitelik Çıkarımı Yapılıyor...", text_color="#F1C40F")
        self.lbl_details.configure(text="")
        self.update()
        
        # Arayüzün çizilmesi için kısa bir gecikme
        self.after(100, self.predict_vehicle)

    def predict_vehicle(self):
        if knn_classifier is None:
            self.lbl_result.configure(text="HATA: Veritabanı Boş", text_color="#E74C3C")
            return

        query_vector = extract_features(self.selected_img_path)
        if query_vector is None: return

        # KNN Sorgusu: En yakın komşuları ve mesafeleri getir
        distances, indices = knn_classifier.kneighbors([query_vector], n_neighbors=N_NEIGHBORS)
        
        neighbors = [stored_labels[i] for i in indices[0]]
        
        # Oylama (En çok tekrar eden sınıfı bul)
        predicted_label = max(set(neighbors), key=neighbors.count)
        vote_count = neighbors.count(predicted_label)
        
        # Ortalama Benzerlik Mesafesi (Cosine Distance: 0=Aynı, 1=Farklı)
        avg_distance = np.mean(distances[0])
        confidence_score = (1 - avg_distance) * 100

        # --- KARAR MEKANİZMASI ---
        debug_info = f"Mesafe: {avg_distance:.4f} (Eşik: {SIMILARITY_THRESHOLD}) | Komşular: {neighbors}"

        # Kural 1: Eşik Değeri Kontrolü
        if avg_distance > SIMILARITY_THRESHOLD:
            self.lbl_result.configure(text="TANIMLANAMAYAN ARAÇ", text_color="#E74C3C")
            self.lbl_details.configure(text=f"Benzerlik oranı çok düşük.\nTahmin: {predicted_label} (?)\n{debug_info}")
            
            if messagebox.askyesno("Veri Girişi", "Bu araç veritabanında tanımlı değil veya güven oranı düşük.\nSisteme yeni eğitim verisi olarak eklemek ister misiniz?"):
                self.add_batch_data()

        # Kural 2: Oy Birliği Kontrolü (Majority Vote)
        elif vote_count < (N_NEIGHBORS / 2):
            self.lbl_result.configure(text=f"SONUÇ BELİRSİZ ({predicted_label}?)", text_color="#E67E22")
            self.lbl_details.configure(text=f"Sınıflandırıcı kararsız kaldı.\n{debug_info}")

        else:
            self.lbl_result.configure(text=f"TESPİT EDİLDİ: {predicted_label}", text_color="#2ECC71")
            self.lbl_details.configure(text=f"Güven Skoru: %{confidence_score:.2f} | Oy: {vote_count}/{N_NEIGHBORS}")

    def add_batch_data(self):
        brand = simpledialog.askstring("Veri Girişi", "Araç Markası:")
        if not brand: return
        model = simpledialog.askstring("Veri Girişi", "Araç Modeli:")
        if not model: return

        # Klasör isimlendirme standardizasyonu
        brand = brand.strip().replace(" ", "_").upper()
        model = model.strip().replace(" ", "_").upper()

        file_paths = filedialog.askopenfilenames(title="Eğitim Görüntülerini Seçiniz")
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
                print(f"Dosya kopyalama hatası: {e}")

        self.refresh_database()
        messagebox.showinfo("İşlem Başarılı", f"{count} adet görüntü veritabanına eklendi:\n{brand} {model}")

if __name__ == "__main__":
    app = CarRecognitionApp()
    app.mainloop()