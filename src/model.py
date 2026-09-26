import torch.nn as nn
import torch.nn.functional as F


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()

        # Blok 1: Görüntüden temel özniteliklerin (kenar, köşe) çıkarılması
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Blok 2: Karmaşık dokuların öğrenilmesi
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Blok 3: Gelişmiş yüz yapıları (gözler, dudaklar)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Tam Bağlantılı (Fully Connected) Katmanlar
        # 48x48 görüntü 3 kez havuzlamadan sonra 6x6 boyutuna düşer
        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.pool1(F.relu(self.bn1_2(self.conv1_2(x))))

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = self.pool2(F.relu(self.bn2_2(self.conv2_2(x))))

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = self.pool3(F.relu(self.bn3_2(self.conv3_2(x))))

        x = x.view(x.size(0), -1)  # Flatten işlemi (Düzleştirme)

        x = self.dropout1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)

        return x