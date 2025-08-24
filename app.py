import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

# -----------------------------
# Load saved objects
# -----------------------------
@st.cache_resource
def load_dependencies():
    model = load_model("second_model.h5")  # your trained captioning model
    
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    return model, tokenizer, 35

model, tokenizer, max_length = load_dependencies()

# -----------------------------
# Feature extractor using VGG16
# -----------------------------
@st.cache_resource
def get_feature_extractor():
    vgg = VGG16()
    vgg = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)  # remove classification head
    return vgg

feature_extractor = get_feature_extractor()

def extract_features(filename, model):
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

# -----------------------------
# Helper functions
# -----------------------------
def index_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final_caption

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🖼️ Image Caption Generator")
st.write("Upload an image and let the AI generate a caption for it.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Extract features
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())
    
    image_features = extract_features("temp.jpg", feature_extractor)
    
    # Generate caption
    caption = predict_caption(model, image_features, tokenizer, max_length)
    
    st.subheader("Generated Caption:")
    st.success(caption)
