# MLPforMNIST_classification
# 🧠 MLP from Scratch - MNIST Classification

Đây là một chương trình huấn luyện **Multilayer Perceptron (MLP)** để phân loại chữ số viết tay từ tập dữ liệu **MNIST**, được cài đặt hoàn toàn bằng **NumPy** (không sử dụng các thư viện deep learning như PyTorch/TensorFlow).

---

## 🧪 Mục tiêu

- Hiểu cách hoạt động của **MLP** và **thuật toán Backpropagation**
- Cài đặt thủ công:
  - Hàm kích hoạt ReLU, Softmax
  - Loss function CrossEntropy
  - Gradient descent cập nhật trọng số
- Huấn luyện model phân loại ảnh MNIST (28x28 → 784 chiều)
- So sánh kết quả giữa model tự viết và PyTorch

---

## 📦 Phụ thuộc

- Python ≥ 3.6
- NumPy
- Matplotlib
- torchvision (để tải MNIST)

```bash
pip install numpy matplotlib torchvision
