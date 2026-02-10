import streamlit as st
import torch
from PIL import Image
import numpy as np
import os


st.write("Current working directory:", os.getcwd())
st.write("All files/folders in current directory:", os.listdir("."))
st.write("Does 'my_photos' exist?", os.path.exists("my_photos"))
if os.path.exists("my_photos"):
    st.write("Files in my_photos:", os.listdir("my_photos"))
else:
    st.write("my_photos NOT found — check repo structure")

st.title("Semantic search with clip model")

# Use a tiny model that has very small weights and loads fast
# This one is RN50 – small, fast, no big download
device = "cpu"

# Tiny CLIP-like model from open-clip (RN50 is small and usually caches quickly)
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='yfcc15m')
model = model.to(device)
tokenizer = open_clip.get_tokenizer('RN50')

# If even this fails, comment above and use this dummy placeholder for testing
# (uncomment if needed)
# class DummyModel:
#     def encode_image(self, x): return torch.randn(x.shape[0], 512)
#     def encode_text(self, x): return torch.randn(x.shape[0], 512)
# model = DummyModel()
# def preprocess(img): return torch.randn(3, 224, 224)
# def tokenize(t): return torch.randint(0, 100, (1, 77))

st.write("Model loaded (small RN50 version)")

# Rest of your code (photos from my_photos folder)
image_folder = "my_photos"
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
               if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

if not image_paths:
    st.error("No photos in 'my_photos' folder. Please add some images.")
    st.stop()

st.write(f"Found {len(image_paths)} photos.")

# Process images
image_embeddings = []
for path in image_paths:
    img = Image.open(path).convert("RGB")
    prep = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(prep).cpu().numpy()
    image_embeddings.append(emb)

st.success("Photos processed!")

query = st.text_input("Search (e.g. 'dog in hands', 'sunset')")

if query:
    text = tokenizer([query]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text).cpu().numpy()

    similarities = [np.dot(text_emb.flatten(), emb.flatten()) for emb in image_embeddings]
    top_idx = np.argmax(similarities)
    best_path = image_paths[top_idx]

    st.write("**Best match:**", os.path.basename(best_path))

    st.image(best_path)
