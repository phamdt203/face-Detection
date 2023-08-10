import cv2
import torch
import pickle
import numpy as np
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from parameters import *
from faceDetector import *
import torch.nn.functional as F
from model import mobilenet_v2
import streamlit as st

def img_to_encoding(image, model, preprocess):
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        return model(image_tensor)

def similarity_score(firstEmbedding, secondEmbedding):
    return F.cosine_similarity(firstEmbedding, secondEmbedding)

def faceDetector(faceDetector, image):    
    image = np.array(image)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceRects = faceDetector.detectMultiScale(gray_img, 1.1, 9)
    face = np.zeros((image.shape[0], image.shape[1]))
    for (x, y, w, h) in faceRects:
        left, top, right, bottom = x, y, x + w, y + h
        face = image[top : bottom, left : right]
    face = Image.fromarray(face)
    return face

def faceRecognition(image, database, model, preprocess):
    fd = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image_cropped = faceDetector(fd, image)
    testEmbedding = img_to_encoding(image_cropped, model, preprocess)
    min_dist = -1
    face_name = ""
    for face in list(database):
        for image in database[face]:
            similarityScore = similarity_score(testEmbedding, image)
            if similarityScore.item() > min_dist:
                min_dist = similarityScore.item()
                face_name = face
    return face_name, image_cropped


def main():
    model = mobilenet_v2()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('facenet_model.pth', map_location=device))
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])

    with open("./path_dict.p", "rb") as f:
        paths = pickle.load(f)

    with open("./database_dict.p", "rb") as f:
        database = pickle.load(f)

    st.title("Face Recognition App")
    st.write("Upload an image for face recognition")
    
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Recognize"):
            result, image_cropped = faceRecognition(image, database, model, preprocess)
            st.write("Recognition Result:", result)
            st.image(image_cropped, caption = "cropped image", use_column_width=True)
    
if __name__ == "__main__":
    main()