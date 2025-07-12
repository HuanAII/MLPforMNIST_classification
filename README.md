# 🧠 MLPforMNIST_classification

Một dự án huấn luyện mô hình **Multilayer Perceptron (MLP)** để phân loại chữ số viết tay từ tập dữ liệu **MNIST**.

Dự án bao gồm:
- Cài đặt MLP từ đầu bằng **NumPy** (không dùng thư viện deep learning)
- Cài đặt MLP bằng **PyTorch** để so sánh
- Triển khai ứng dụng web bằng **Streamlit** cho phép người dùng vẽ chữ số và nhận kết quả dự đoán từ mô hình đã huấn luyện.

---

## 🎯 Mục tiêu

- Hiểu và cài đặt:
  - Hàm kích hoạt: **ReLU**, **Softmax**
  - Hàm mất mát: **CrossEntropy**
  - Thuật toán: **Backpropagation**, **Gradient Descent**
- Huấn luyện model MLP từ ảnh đầu vào 28×28 (784 chiều)
- So sánh hiệu quả mô hình viết tay và PyTorch
- Triển khai dự đoán trực tuyến bằng Streamlit

---

## 📁 Cấu trúc dự án

```bash
MLPforMNIST_classification/
├── mlp_numpy.py               # MLP viết tay bằng NumPy
├── train_numpy.py             # Huấn luyện mô hình NumPy
├── mlp_pytorch.py             # Định nghĩa MLP bằng PyTorch
├── train_pytorch.py           # Huấn luyện và lưu PyTorch model
├── mlp_mnist.pt               # Trọng số PyTorch đã huấn luyện
├── streamlit_app.py           # Ứng dụng Streamlit vẽ và dự đoán số
├── README.md


-Yêu cầu Python ≥ 3.6
- Cài đặt các thư viện cần thiết:
pip install numpy matplotlib torchvision torch streamlit streamlit-drawable-canvas Pillow


