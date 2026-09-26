import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import torchvision.transforms as transforms
from collections import deque
import threading

# Tema ayarları
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()

        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(0.25)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.25)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.drop_fc2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        x = x.view(x.size(0), -1)
        x = self.drop_fc1(self.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop_fc2(self.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        return x


EMOTION_LABELS = ['Kızgın', 'İğrenme', 'Korku', 'Mutlu', 'Üzgün', 'Şaşkın', 'Doğal']


def load_model(model_path="best_emotion_model.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = EmotionCNN(num_classes=len(EMOTION_LABELS))
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        print(f"\n[HATA] Model yüklenemedi: {e}\n")
        return None, device


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Kararlı ve Hızlı Duygu Analizi")
        self.geometry("1050x680")
        self.resizable(False, False)

        self.model, self.device = load_model()

        # OpenCV Yüz Algılama (MediaPipe kullanmıyoruz, sorunsuz çalışır)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.cap = None
        self.is_camera_running = False
        self.current_probabilities = np.zeros(len(EMOTION_LABELS))
        # Titremeyi engellemek için son 10 kareyi hafızada tutar
        self.prediction_history = deque(maxlen=10)

        self.setup_ui()

    def setup_ui(self):
        left_frame = ctk.CTkFrame(self, fg_color="transparent")
        left_frame.pack(side="left", padx=20, pady=20, fill="both", expand=True)

        title_label = ctk.CTkLabel(left_frame, text="Yüksek Doğruluklu Duygu Analizi",
                                   font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=(0, 10))

        self.video_label = ctk.CTkLabel(left_frame, text="Görüntü Yükleyin veya Kamerayı Açın", fg_color="#1a1a1a",
                                        corner_radius=10, width=640, height=480)
        self.video_label.pack()

        btn_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        btn_frame.pack(pady=15, fill="x")

        self.btn_image = ctk.CTkButton(btn_frame, text="📁 Resim Yükle", font=ctk.CTkFont(size=14, weight="bold"),
                                       command=self.load_image, height=40, fg_color="#2980b9", hover_color="#3498db")
        self.btn_image.pack(side="left", padx=10, expand=True, fill="x")

        self.btn_cam = ctk.CTkButton(btn_frame, text="📷 Kamerayı Başlat", font=ctk.CTkFont(size=14, weight="bold"),
                                     command=self.toggle_camera, height=40, fg_color="#c0392b", hover_color="#e74c3c")
        self.btn_cam.pack(side="left", padx=10, expand=True, fill="x")

        right_frame = ctk.CTkFrame(self, width=320, corner_radius=15)
        right_frame.pack(side="right", padx=20, pady=20, fill="y")
        right_frame.pack_propagate(False)

        analysis_title = ctk.CTkLabel(right_frame, text="Duygu Dağılımı (Filtreli)",
                                      font=ctk.CTkFont(size=16, weight="bold"))
        analysis_title.pack(pady=15)

        self.bars = {}
        self.bar_labels = {}

        for emotion in EMOTION_LABELS:
            row_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
            row_frame.pack(pady=6, padx=15, fill="x")

            lbl = ctk.CTkLabel(row_frame, text=f"{emotion}: %0", font=ctk.CTkFont(size=12, weight="bold"), width=90,
                               anchor="w")
            lbl.pack(side="left")

            bar = ctk.CTkProgressBar(row_frame, orientation="horizontal", height=12)
            bar.set(0.0)
            bar.pack(side="right", expand=True, fill="x", padx=(5, 0))

            self.bar_labels[emotion] = lbl
            self.bars[emotion] = bar

    def detect_and_predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        dominant_emotion = "Yüz Bulunamadı"

        for (x, y, w, h) in faces:
            # Yüz kırpmayı genişleterek kaliteyi artırma
            margin = int(w * 0.1)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)

            roi_gray = gray[y1:y2, x1:x2]

            if self.model is not None and roi_gray.size > 0:
                try:
                    tensor = transform(roi_gray).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(tensor)
                        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

                        self.prediction_history.append(probs)
                        # Son 10 karenin ortalamasını al (Titremeyi %100 çözer)
                        smoothed_probs = np.mean(self.prediction_history, axis=0)
                        self.current_probabilities = smoothed_probs

                        pred_idx = np.argmax(smoothed_probs)
                        dominant_emotion = EMOTION_LABELS[pred_idx]
                        confidence = smoothed_probs[pred_idx] * 100
                        dominant_emotion = f"{dominant_emotion} (%{confidence:.1f})"
                except Exception as ex:
                    dominant_emotion = "Analiz Hatası"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (46, 204, 113), 3)
            cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (46, 204, 113), 2)
            break

        if len(faces) == 0:
            self.current_probabilities = np.zeros(len(EMOTION_LABELS))
            self.prediction_history.clear()

        return frame

    def update_gui_bars(self):
        for i, emotion in enumerate(EMOTION_LABELS):
            prob = float(self.current_probabilities[i])
            self.bars[emotion].set(prob)
            self.bar_labels[emotion].configure(text=f"{emotion}: %{int(prob * 100)}")

    def load_image(self):
        self.stop_camera()
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

        if file_path:
            self.prediction_history.clear()
            frame = cv2.imread(file_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.detect_and_predict(frame)
            self.update_gui_bars()
            self.display_frame(frame)

    def toggle_camera(self):
        if not self.is_camera_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Hata", "Kamera başlatılamadı!")
                return

            self.prediction_history.clear()
            self.is_camera_running = True
            self.btn_cam.configure(text="Kamerayı Kapat", fg_color="#7f8c8d", hover_color="#95a5a6")

            self.thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.thread.start()
        else:
            self.stop_camera()

    def camera_loop(self):
        while self.is_camera_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.detect_and_predict(frame)
                self.after(0, self.update_gui_bars)
                self.after(0, self.display_frame, frame)

    def stop_camera(self):
        self.is_camera_running = False
        self.btn_cam.configure(text="Kamerayı Başlat", fg_color="#c0392b", hover_color="#e74c3c")
        if self.cap:
            self.cap.release()
        self.current_probabilities = np.zeros(len(EMOTION_LABELS))
        self.prediction_history.clear()
        self.update_gui_bars()

    def display_frame(self, frame):
        image = Image.fromarray(frame)
        ctk_image = ctk.CTkImage(light_image=image, dark_image=image, size=(640, 480))
        self.video_label.configure(image=ctk_image, text="")
        self.video_label.image = ctk_image


if __name__ == "__main__":
    app = EmotionApp()
    app.mainloop()