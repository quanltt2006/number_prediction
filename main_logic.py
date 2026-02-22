import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 1. Cài đặt thư mục và thiết bị
ROOT = "./data"
SAVE_DIR = './model'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load dữ liệu MNIST
train_data = datasets.MNIST(root=ROOT, train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root=ROOT, train=False, download=True, transform=transforms.ToTensor())

# Chia tập Validation
VALID_RATIO = 0.9
n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples
train_data, valid_data = data.random_split(train_data, [n_train_examples, n_valid_examples])

# Tính mean/std để chuẩn hóa
mean = 0.1307 # Giá trị chuẩn của MNIST
std = 0.3081

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
])

# Gán transform (Lưu ý: transform cần gán cho dataset gốc hoặc qua Subset)
train_data.dataset.transform = train_transform
valid_data.dataset.transform = train_transform

BATCH_SIZE = 128 
train_dataloader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = data.DataLoader(valid_data, batch_size=BATCH_SIZE)

# 3. Định nghĩa Model LeNet-5
class LeNetClassifer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # C1: 28x28 -> 6 feature maps 28x28 (padding=2 để giữ size)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        
        # C3: 14x14 -> 16 feature maps 10x10
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        
        # F5, F6, Out
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.avgpool1(self.conv1(x)))
        x = F.relu(self.avgpool2(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x

# 4. Hàm Train
def train(model, optimizer, criterion, dataloader, device, epoch, log_interval=50):
    model.train()
    total_acc, total_count = 0, 0
    losses = []
    
    # Biến để logging
    log_acc, log_count = 0, 0
    start_time = time.time()

    for idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        # Tính toán accuracy
        acc = (predictions.argmax(1) == labels).sum().item()
        total_acc += acc
        total_count += labels.size(0)
        log_acc += acc
        log_count += labels.size(0)
        losses.append(loss.item())

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(f"| epoch {epoch:3d} | {idx:5d}/{len(dataloader):5d} batches | accuracy {log_acc / log_count:8.3f}")
            log_acc, log_count = 0, 0
            start_time = time.time()
            
    return total_acc / total_count, sum(losses) / len(losses)

# 5. Hàm Evaluate
def evaluate(model, criterion, dataloader, device):
    model.eval()
    total_acc, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            
            losses.append(loss.item())
            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
            
    return total_acc / total_count, sum(losses) / len(losses)

# 6. Khởi tạo Training
num_classes = 10
lenet_model = LeNetClassifer(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet_model.parameters(), lr=0.001)

num_epochs = 20
best_loss_eval = float('inf')

# Vòng lặp Training
for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()

    train_acc, train_loss = train(lenet_model, optimizer, criterion, train_dataloader, device, epoch)
    eval_acc, eval_loss = evaluate(lenet_model, criterion, valid_dataloader, device)

    # Lưu model tốt nhất
    if eval_loss < best_loss_eval:
        best_loss_eval = eval_loss
        torch.save(lenet_model.state_dict(), os.path.join(SAVE_DIR, 'lenet_model.pt'))
        print(f"--> Saved best model with loss: {best_loss_eval:.4f}")

    print(f"Epoch {epoch:3d} | train acc {train_acc:8.3f} | train loss {train_loss:8.3f} "
          f"| eval acc {eval_acc:8.3f} | eval loss {eval_loss:8.3f} "
          f"| time {(time.time() - epoch_start_time):5.2f}s")
    print("-" * 80)

print("Training hoàn tất!")

# 7. (Tùy chọn) Load model tốt nhất để test cuối cùng
lenet_model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'lenet_model.pt')))
test_dataloader = data.DataLoader(test_data, batch_size=BATCH_SIZE)
final_acc, final_loss = evaluate(lenet_model, criterion, test_dataloader, device)
print(f"KẾT QUẢ TRÊN TẬP TEST: Accuracy: {final_acc:.4f}, Loss: {final_loss:.4f}")
