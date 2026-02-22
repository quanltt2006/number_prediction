import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# 1. Định nghĩa lại cấu trúc Model (Phải giống hệt file code.py của bạn)
class LeNetClassifer(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
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

# 2. Hàm load model đã train
@st.cache_resource
def load_model():
    model = LeNetClassifer(num_classes=10)
    # Đường dẫn này trỏ vào thư mục model/ file lenet_model.pt của bạn
    model_path = 'model/lenet_model.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        st.error(f"Không tìm thấy file model tại {model_path}")
        return None

model = load_model()

# 3. Giao diện Streamlit
st.set_page_config(page_title="MNIST Digit Predictor", layout="centered")
st.title("🔢 Dự đoán chữ số viết tay LeNet-5")
st.write("Dự án CNN 2026 - Nhận diện chữ số từ bộ dữ liệu MNIST")

uploaded_file = st.file_uploader("Upload ảnh chữ số (nền đen chữ trắng sẽ chính xác hơn)...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L') # Chuyển về ảnh xám (Grayscale)
    st.image(image, caption='Ảnh đã upload', width=200)
    
    # 4. Tiền xử lý ảnh (Dùng đúng thông số mean/std bạn đã dùng lúc train)
    mean = 0.1307
    std = 0.3081
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])
    
    img_tensor = transform(image).unsqueeze(0) # Thêm batch dimension

    # 5. Dự đoán
    if st.button('Kiểm tra kết quả'):
        if model is not None:
            with torch.no_grad():
                output = model(img_tensor)
                # Tính xác suất bằng Softmax
                probabilities = F.softmax(output, dim=1)
                prob, predicted = torch.max(probabilities, 1)
                
                st.success(f"### Kết quả dự đoán: {predicted.item()}")
                st.write(f"Độ tin cậy: {prob.item()*100:.2f}%")
