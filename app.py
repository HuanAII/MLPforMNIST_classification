import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from streamlit_drawable_canvas import st_canvas
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)


model = MLP(784, 128, 10)
model.load_state_dict(torch.load("mlp_mnist.pt", map_location=torch.device('cpu')))
model.eval()


st.title(" MNIST Digit Recognition with MLP")
st.write(" Vẽ một chữ số (0-9) bên dưới:")

canvas_result = st_canvas(
    fill_color="#000000",  # màu nền nét vẽ (đen)
    stroke_width=10,
    stroke_color="#FFFFFF",  # màu nét vẽ (trắng)
    background_color="#000000",  # nền đen
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# ✅ Dự đoán khi nhấn nút
if st.button(" Dự đoán"):
    if canvas_result.image_data is not None:
        # Lấy ảnh người dùng vẽ
        img = canvas_result.image_data
        img = Image.fromarray((img[:, :, 0]).astype(np.uint8))  # chỉ lấy kênh R
        img = ImageOps.invert(img)  # đảo màu trắng đen
        img = img.resize((28, 28)).convert("L")  # resize và chuyển ảnh xám

        # Tiền xử lý cho model
        img_tensor = transforms.ToTensor()(img)
        img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)
        img_tensor = img_tensor.view(1, -1)

        # Dự đoán
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()

        st.image(img.resize((140, 140)), caption=" Ảnh đã xử lý")
        st.success(f"✅ Mô hình dự đoán: **{pred}**")
    else:
        st.warning(" Bạn cần vẽ một chữ số trước.")
