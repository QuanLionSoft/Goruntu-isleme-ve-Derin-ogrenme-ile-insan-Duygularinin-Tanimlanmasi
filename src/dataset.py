import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(data_dir, batch_size=64):
    """
    Klasör yapısındaki görüntüleri okur (örneğin: data/train/angry/001.jpg)
    """

    # Eğitim verisi için Augmentation (Veri Çoğaltma)
    train_transform = transforms.Compose([
        # RGB okunan resimleri modelimize uygun tek kanallı Gri Tonlamaya çeviriyoruz
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((48, 48)),  # Boyutları 48x48 garantiye alıyoruz
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Test verisinde sadece boyutlandırma ve normalizasyon yapılır
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Klasörlerden veri setlerini oluştur
    # Klasör isimleri otomatik olarak etiket (label) kabul edilir
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Sınıf isimlerini (angry, happy, sad vb.) döndürüyoruz
    return train_loader, val_loader, train_dataset.classes