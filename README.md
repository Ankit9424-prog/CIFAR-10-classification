# CIFAR-10 Image Classification | PyTorch CNN (From Scratch)

An end-to-end **deep learning project** implementing a **Convolutional Neural Network (CNN)** from scratch using **PyTorch** for **image classification** on the **CIFAR-10 dataset**.

🔹 Designed to demonstrate **core machine learning and deep learning fundamentals**  
🔹 Built without pre-trained models to show **conceptual understanding**

---

## 🎯 Project Objective

To design, train, and evaluate a CNN capable of classifying 32×32 RGB images into 10 categories using a fully custom PyTorch pipeline.

---

## 🧠 Technical Skills Demonstrated

- PyTorch (model building, training, inference)
- Convolutional Neural Networks (CNNs)
- Dataset handling with `torchvision`
- GPU acceleration (CUDA support)
- Training & evaluation loops
- Model performance analysis
- Confusion matrix visualization
- Clean, modular deep learning code

---

## 🏗️ Model Architecture

- **Input:** 3 × 32 × 32 RGB images
- **Convolution Block 1:**  
  Conv2D → ReLU → Conv2D → ReLU → MaxPooling
- **Convolution Block 2:**  
  Conv2D → ReLU → Conv2D → ReLU → MaxPooling
- **Classifier:**  
  Flatten → Fully Connected Layer
- **Output:** 10 class logits

---

## 🗂️ Repository Structure
```
├── cifar10.py 
├── CIFAR10.ipynb #(recommended)
└── README.md 
```


---

## 🚀 How to Run (Recommended)

**Google Colab** is recommended for easy setup and GPU access.

1. Open `CIFAR10.ipynb` in Google Colab  
2. Enable GPU:  
   `Runtime → Change runtime type → GPU`
3. Run all cells sequentially

The CIFAR-10 dataset is downloaded automatically.

---

## 📊 Evaluation & Results

- Training and testing loss per epoch
- Training and testing accuracy per epoch
- Visual predictions on unseen test images
- Confusion matrix for class-wise performance analysis

---

## 🧪 Key Machine Learning Concepts Applied

- Supervised learning
- Multi-class classification
- Backpropagation
- Optimization using Adam
- Overfitting awareness via train/test split
- Model evaluation beyond accuracy

---

## 🔍 Why This Project Matters for ML Internships

This project demonstrates:
- Ability to **build neural networks from first principles**
- Strong understanding of **PyTorch workflow**
- Hands-on experience with **real-world image data**
- Clean and readable ML code
- Practical evaluation and visualization skills

---

## 👤 Author

**Ankit**  
Machine Learning & Computer Science Student  
Actively building end-to-end ML & Deep Learning projects with PyTorch

---

## ⭐ Acknowledgements

- CIFAR-10 Dataset
- PyTorch & Torchvision
- Google Colab


