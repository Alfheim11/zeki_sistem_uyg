import os
import numpy as np
import threading
import datetime
import shutil
import cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image

# --- TensorFlow Ayarları ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# --- SİSTEM AYARLARI ---
DATABASE_DIR = "vehicle_database"
MODEL_FILE = "neurocar_model.keras" 
IMG_SIZE = (224, 224) 
BATCH_SIZE = 8       
EPOCHS = 40          #(Daha uzun sürer ama daha iyi öğrenir)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class NeuroCarApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("NeuroCar - AI Vehicle Recognition System")
        self.geometry("1100x850")
        
        self.model = None
        self.class_names = [] 
        self.is_training = False

        # --- BAŞLIK ---
        self.lbl_head = ctk.CTkLabel(self, text="NEUROCAR | DERİN ÖĞRENME ANALİZ SİSTEMİ", font=("Segoe UI", 26, "bold"))
        self.lbl_head.pack(pady=20)

        # --- GÖRSEL PANELİ ---
        self.img_panel = ctk.CTkLabel(
            self, 
            text="Sistem Hazır.\nAnaliz için görüntü yükleyiniz.", 
            width=800, height=450, 
            fg_color="#2b2b2b", 
            corner_radius=15
        )
        self.img_panel.pack(pady=10)

        # --- DURUM PANELİ ---
        self.lbl_status = ctk.CTkLabel(self, text="Model durumu bekleniyor...", font=("Segoe UI", 16), text_color="#bdc3c7")
        self.lbl_status.pack(pady=5)
        
        self.progressbar = ctk.CTkProgressBar(self, width=600, mode="indeterminate")
        self.progressbar.pack(pady=5)
        self.progressbar.stop() 

        # --- KONTROL PANELİ ---
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=20)

        # Butonlar
        self.btn_predict = ctk.CTkButton(
            btn_frame, text="GÖRÜNTÜ ANALİZİ YAP", 
            command=self.predict_image, 
            width=220, height=50, 
            font=("Segoe UI", 13, "bold"),
            fg_color="#3498db", hover_color="#2980b9"
        )
        self.btn_predict.pack(side="left", padx=15)

        self.btn_add = ctk.CTkButton(
            btn_frame, text="VERİ SETİNE EKLE", 
            command=self.add_data, 
            width=180, height=50, 
            font=("Segoe UI", 13, "bold"),
            fg_color="#e67e22", hover_color="#d35400"
        )
        self.btn_add.pack(side="left", padx=15)
        
        self.btn_train = ctk.CTkButton(
            btn_frame, text="MODEL EĞİTİMİNİ BAŞLAT", 
            command=self.start_training, 
            width=220, height=50, 
            font=("Segoe UI", 13, "bold"),
            fg_color="#27ae60", hover_color="#2ecc71"
        )
        self.btn_train.pack(side="left", padx=15)

        # Başlangıç Yüklemesi
        threading.Thread(target=self.load_trained_model, daemon=True).start()

    def load_trained_model(self):
        if os.path.exists(MODEL_FILE):
            self.lbl_status.configure(text="Sistem Başlatılıyor: Model Yükleniyor...", text_color="#f1c40f")
            try:
                self.model = load_model(MODEL_FILE)
                if os.path.exists(DATABASE_DIR):
                    self.class_names = sorted([d for d in os.listdir(DATABASE_DIR) if os.path.isdir(os.path.join(DATABASE_DIR, d))])
                self.lbl_status.configure(text=f"SİSTEM AKTİF | {len(self.class_names)} Sınıf Tanımlı", text_color="#2ecc71")
            except Exception as e:
                self.lbl_status.configure(text=f"Model Hatası: {e}", text_color="#e74c3c")
        else:
            self.lbl_status.configure(text="Eğitilmiş model yok. Eğitim bekleniyor.", text_color="#e74c3c")

    def add_data(self):
        marka = simpledialog.askstring("Veri Girişi", "Araç Marka/Model (Örn: BMW_M5):")
        if not marka: return
        marka = marka.strip().replace(" ", "_").upper()
        
        files = filedialog.askopenfilenames(title="Eğitim Görsellerini Seçiniz")
        if not files: return
        
        target_dir = os.path.join(DATABASE_DIR, marka)
        os.makedirs(target_dir, exist_ok=True)
        
        count = 0
        for f in files:
            try:
                ext = os.path.splitext(f)[1]
                if ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']: continue
                new_name = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{count}{ext}"
                shutil.copy(f, os.path.join(target_dir, new_name))
                count += 1
            except: pass
            
        messagebox.showinfo("Tamamlandı", f"{count} görsel eklendi. Lütfen modeli eğitin.")
        self.lbl_status.configure(text="Veri seti güncellendi. Yeniden eğitim gerekli.", text_color="#f39c12")

    def start_training(self):
        if self.is_training: return
        if not os.path.exists(DATABASE_DIR) or len(os.listdir(DATABASE_DIR)) < 2:
            messagebox.showwarning("Yetersiz Veri", "Eğitim için en az 2 farklı araç sınıfı gereklidir.")
            return

        self.is_training = True
        self.btn_train.configure(state="disabled", text="Eğitim Sürüyor (Bekleyiniz)...")
        self.progressbar.start()
        
        # Eğitimi thread içinde başlat
        threading.Thread(target=self.train_cnn_model, daemon=True).start()

    def train_cnn_model(self):
        try:
            self.lbl_status.configure(text="Veri seti işleniyor ve artırılıyor...", text_color="#3498db")
            
            # Veri Artırma
            train_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
                shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest',
                validation_split=0.2
            )

            train_generator = train_datagen.flow_from_directory(
                DATABASE_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                class_mode='categorical', subset='training'
            )
            
            validation_generator = train_datagen.flow_from_directory(
                DATABASE_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                class_mode='categorical', subset='validation'
            )

            num_classes = len(train_generator.class_indices)
            self.class_names = list(train_generator.class_indices.keys())

            self.lbl_status.configure(text="GoogLeNet mimarisi kuruluyor...", text_color="#3498db")
            
            # Model Mimarisi
            base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x) 
            predictions = Dense(num_classes, activation='softmax')(x)

            model = Model(inputs=base_model.input, outputs=predictions)
            for layer in base_model.layers: layer.trainable = False

            model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

            self.lbl_status.configure(text=f"Eğitim Başladı ({EPOCHS} Epoch)... Bu işlem uzun sürecektir.", text_color="#f1c40f")
            
            # --- EĞİTİM ---
            history = model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=EPOCHS
            )

            self.lbl_status.configure(text="Model kaydediliyor...", text_color="#2ecc71")
            model.save(MODEL_FILE)
            self.model = model
            
            final_acc = history.history['accuracy'][-1] * 100
            self.lbl_status.configure(text=f"Eğitim Bitti | Başarı: %{final_acc:.1f}", text_color="#2ecc71")
            messagebox.showinfo("Başarılı", f"Eğitim tamamlandı!\nModel Başarısı: %{final_acc:.1f}")

        except Exception as e:
            print(f"Hata: {e}")
            self.lbl_status.configure(text="Eğitim hatası!", text_color="#e74c3c")
            messagebox.showerror("Hata", str(e))
        
        finally:
            self.progressbar.stop()
            self.is_training = False
            self.btn_train.configure(state="normal", text="MODEL EĞİTİMİNİ BAŞLAT")

    def predict_image(self):
        if self.model is None:
            messagebox.showwarning("Model Bulunamadı", "Lütfen önce model eğitimini tamamlayınız.")
            return

        file_path = filedialog.askopenfilename()
        if not file_path: return

        try:
            # Görüntü Yükleme ve Ön İşleme
            img_original = Image.open(file_path)
            
            # PNG/Şeffaflık Düzeltmesi
            if img_original.mode != 'RGB':
                img_original = img_original.convert('RGB')

            # Arayüzde Gösterim
            ratio = min(800/img_original.width, 450/img_original.height)
            new_size = (int(img_original.width*ratio), int(img_original.height*ratio))
            ctk_img = ctk.CTkImage(img_original, size=new_size)
            self.img_panel.configure(image=ctk_img, text="")

            # Modele Hazırlık
            img = img_original.resize(IMG_SIZE)
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Tahmin
            preds = self.model.predict(x)
            score = np.max(preds)
            class_idx = np.argmax(preds)
            
            if class_idx < len(self.class_names):
                class_name = self.class_names[class_idx].replace("_", " ")
                confidence_val = score * 100
                
                if score > 0.6:
                    res_text = f"TESPİT EDİLDİ: {class_name}"
                    color = "#2ecc71"
                else:
                    res_text = f"OLASI EŞLEŞME: {class_name} (?)"
                    color = "#e67e22"
                
                status_text = f"{res_text} | Güven Oranı: %{confidence_val:.2f}"
            else:
                status_text = "Tanımlanamayan Araç"
                color = "#e74c3c"

            self.lbl_status.configure(text=status_text, text_color=color)
            print(f"Tahmin Skoru: {preds}")

        except Exception as e:
            self.lbl_status.configure(text=f"Analiz Hatası: {e}", text_color="#e74c3c")
            messagebox.showerror("Hata", str(e))

if __name__ == "__main__":
    app = NeuroCarApp()

    app.mainloop()
