import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_model(model, train_loader, val_loader, epochs=30, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Akademik standart: Adam optimizer ve Learning Rate Scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch + 1}/{epochs}")
        # TQDM ile profesyonel progress bar
        train_bar = tqdm(train_loader, desc="Eğitim")

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Gradientleri sıfırla
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()  # Backpropagation
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            train_bar.set_postfix({'Loss': running_loss / total, 'Acc': 100. * correct / total})

        # Validasyon Fazı
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        print(f"Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

        # Dinamik öğrenme oranı ayarı
        scheduler.step(avg_val_loss)

        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_emotion_model.pth')
            print("Yeni en iyi model kaydedildi!")