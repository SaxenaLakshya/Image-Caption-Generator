# 🖼️ Image Caption Generator

A deep learning project that generates captions for images using **VGG16** for feature extraction and **LSTM** for sequence generation.  
Trained on the **Flickr8k** dataset, this model learns to describe images in natural language.

---

## 📌 Features
- **VGG16** pretrained on ImageNet for image feature extraction.
- **LSTM** network for sequence-to-sequence caption generation.
- Tokenization and padding for text preprocessing.
- BLEU score evaluation for model performance.
- Modular code structure for easy extension to other architectures (e.g., InceptionV3).
- Easily adaptable into an API (future plan with FastAPI).

---

## 🛠️ Tech Stack

**Language:**  
- Python 🐍  

**Frameworks & Libraries:**  
- [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/)
- [NumPy](https://numpy.org/)
- [tqdm](https://tqdm.github.io/)
- [NLTK](https://www.nltk.org/)
- [Pillow (PIL)](#)
- [Matplotlib](https://matplotlib.org/)

---

## 📂 Dataset
- **Flickr8k Dataset**
- 8,000 images, each with five captions.
- Captions are preprocessed for tokenization and vocabulary creation.

---

## 📜 How It Works
1. **Image Feature Extraction**:  
   - Images are resized and passed through **VGG16** to extract 4,096-dimensional feature vectors.
2. **Text Preprocessing**:  
   - Tokenization, padding, and integer encoding of captions.
3. **Model Training**:  
   - Image features + partial captions → Predict next word.
4. **Caption Generation**:  
   - Greedy search or beam search for generating complete captions.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/image-caption-generator.git
cd image-caption-generator
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
requirements.txt
```bash
tensorflow
numpy
tqdm
nltk
pillow
matplotlib
```

### 3️⃣ Download Dataset
- Download the **Flickr8k Dataset** (images) from:  
  [Flickr8k_Dataset.zip](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)  
  and extract it into a folder named `dataset/images/`

- Download the **Flickr8k Captions** file from:  
  [Flickr8k_text.zip](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)  
  and extract it into a folder named `dataset/captions/`

**Folder Structure Example:**
```bash
dataset/
│── images/
│   ├── 1000268201_693b08cb0e.jpg
│   ├── 1001773457_577c3a7d70.jpg
│   └── ...
│
└── captions/
    ├── Flickr8k.token.txt
    ├── Flickr8k.trainImages.txt
    ├── Flickr8k.devImages.txt
    ├── Flickr8k.testImages.txt
    └── ...
```

---

## 🔮 Future Improvements

- **Model Enhancements**  
  - Experiment with more advanced CNNs like **InceptionV3**, **ResNet50**, or **EfficientNet** for better feature extraction.  
  - Use transformer-based architectures (e.g., **Vision Transformer (ViT)** + **GPT-style decoder**) for improved caption quality.  

- **API Integration**  
  - Build a **FastAPI** or **Flask** backend to serve trained models.  
  - Allow selection between multiple trained models (VGG16, InceptionV3, etc.) via API endpoints.  

- **User Interface**  
  - Create a web-based frontend with **React.js** or **Streamlit** for easy image uploads and caption display.  

- **Search & Retrieval**  
  - Implement reverse image search using the extracted features.  
  - Add keyword-based search for images using generated captions.  

- **Performance Optimization**  
  - Experiment with **beam search** instead of greedy search for higher-quality captions.  
  - Use mixed-precision training and model quantization for faster inference.  

- **Dataset Expansion**  
  - Train on larger datasets like **Flickr30k** or **MS COCO** for richer vocabulary and better generalization.  

- **Evaluation Metrics**  
  - Include advanced metrics like **METEOR**, **ROUGE-L**, and **CIDEr** for more comprehensive evaluation.
