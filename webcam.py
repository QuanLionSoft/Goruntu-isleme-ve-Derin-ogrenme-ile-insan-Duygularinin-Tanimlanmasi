import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.model import EmotionCNN


def main():
    # 1. Cihaz ve Model Hazırlığı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan donanım: {device}")

    # Sınıflar (Dataset'in tespit ettiği alfabetik sıraya göre)
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # Modeli başlat ve eğitilmiş ağırlıkları yükle
    model = EmotionCNN(num_classes=len(classes))
    try:
        model.load_state_dict(torch.load('best_emotion_model.pth', map_location=device))
        print("Eğitilmiş model (best_emotion_model.pth) başarıyla yüklendi!")
    except FileNotFoundError:
        print("HATA: 'best_emotion_model.pth' bulunamadı. Lütfen önce eğitimin bitmesini bekleyin.")
        return

    model.to(device)
    model.eval()  # Modeli 'tahmin' moduna (değerlendirme) alıyoruz

    # Görüntü Ön İşleme (Sadece Gri tonlama, boyutlandırma ve normalizasyon)
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 2. Yüz Tespiti için OpenCV Haar Cascade Yüklemesi
    # OpenCV'nin kendi içindeki hazır yüz bulma algoritmasını çekiyoruz
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 3. Web Kamerasını Başlat
    cap = cv2.VideoCapture(0)  # 0 varsayılan web kamerasını temsil eder

    print("Kamera açılıyor... Çıkmak için klavyeden 'q' tuşuna basın.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kameradan görüntü alınamadı!")
            break

        # Modeli gri tonlamalı resimlerle eğittik, bu yüzden kamerayı griye çeviriyoruz
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Ekrandaki yüzleri tespit et
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Tespit edilen yüzün etrafına bir dikdörtgen çiz
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Sadece yüz bölgesini kırp
            roi_gray = gray_frame[y:y + h, x:x + w]

            # Kırpılan yüzü PIL formatına çevir ve PyTorch Tensörüne dönüştür
            pil_img = Image.fromarray(roi_gray)
            img_tensor = transform(pil_img).unsqueeze(0).to(device)  # .unsqueeze(0) ile Batch boyutu ekliyoruz

            # 4. Modeli Kullanarak Duygu Tahmini Yap
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

                emotion = classes[predicted.item()]
                conf_score = confidence.item() * 100  # Yüzdelik güven oranı

            # Duyguyu ve güven oranını ekrana yazdır (Örn: happy (%95.2))
            text = f"{emotion} ({conf_score:.1f}%)"

            # Yazı rengini duyguya göre değiştir (Opsiyonel ama şık durur)
            color = (0, 255, 0) if emotion == 'happy' else (0, 0, 255) if emotion == 'angry' else (255, 255, 0)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Sonuç karesini ekranda göster
        cv2.imshow('Gercek Zamanli Duygu Analizi', frame)

        # 'q' tuşuna basılırsa döngüyü kır ve kamerayı kapat
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()