import torch
import os
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image

def preprocessing_image(imageFolderPath):
    images_preprocess = []
    for image_name in os.listdir(imageFolderPath):
        img = cv2.imread(os.path.join(imageFolderPath,image_name ))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) # convert to RGB (torch type)
        img /= 255.0
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = torch.FloatTensor(img).permute(2,0,1) # torch channel
        images_preprocess.append(img)
    return images_preprocess    

class MyDataset(Dataset):
    def __init__(self, imgs, labels, transform =None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        image = self.imgs[index]
        label = self.labels[index]
        return image, label

class CreateDataLoader(DataLoader):
    def __init__(self, dataset, batch_size = 64, shuffle = True):
        super(self, CreateDataLoader).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    def create(self):
        data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle) 
        return data_loader


class EfficientNetClassification():
    def __init__(self, weight_path=None, device = 'cuda'):
        self.device = device
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
        self.model.to(device)
        if weight_path:
            self.model.load_state_dict(torch.load(weight_path, weights_only=True))
    def train(self, num_epochs, optimizer, criterion, train_loader):
        for epoch in range(num_epochs):
            train_loss = 0
            self.model.train()
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = self.model(images)
                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader)}')
    def evaluation(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the model on the validation set: {100 * correct / total:.2f}%')
    def predict(self, img):
        preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize ảnh về kích thước (256, 256)
        transforms.ToTensor(),          # Chuyển ảnh về tensor [C, H, W] và scale giá trị vào khoảng [0, 1]
        ])

        img = preprocess(img).unsqueeze(0)  # Thêm batch dimension
        img = img.to(self.device)
        result = self.model(img) 
        result = torch.softmax(result, -1)

        result = torch.argmax(result).item()
        return result
    
# # Test
if __name__ == '__main__': 
    
    model = EfficientNetClassification(
        weight_path = 'weight/weightinvoiced_weight.pth',
        device = 'cpu'
    )
    # model.load_weight(weight_path=weight)
    folder = 'Dataset/Invoice_Segmentation-7/test'
    for i,img_name in enumerate(os.listdir(folder)):
        if i > 5: break
        path = os.path.join(folder, img_name)
        img = Image.open(path)
        res = model.predict(img)
        print(res)