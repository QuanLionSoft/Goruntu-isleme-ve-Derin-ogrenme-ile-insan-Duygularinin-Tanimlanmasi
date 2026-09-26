import torch
from src.dataset import get_dataloaders
from src.model import EmotionCNN
from src.train import train_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan donanım: {device}")

    # Artık csv_path yerine train ve test klasörlerini içeren ana klasör yolunu veriyoruz
    data_dir = 'data'

    print("Veri setleri hazırlanıyor (Klasör mimarisi kullanılıyor)...")
    train_loader, val_loader, classes = get_dataloaders(data_dir, batch_size=64)

    print(f"Sistem şu sınıfları otomatik tespit etti: {classes}")

    print("Model başlatılıyor...")
    # Sınıf sayısını otomatik olarak tespit edilen klasör sayısına göre ayarlıyoruz
    model = EmotionCNN(num_classes=len(classes))

    print("Eğitim süreci başlıyor...")
    train_model(model, train_loader, val_loader, epochs=30, device=device)


if __name__ == '__main__':
    main()