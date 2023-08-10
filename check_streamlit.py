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
    image = Image.fromarray(image).convert("RGB")
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
    return face

def faceRecognition(frame, database, model, preprocess):
    fd = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    rgb_frame = frame[:, :, ::-1]
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    faces = fd.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for x,y,w,h in faces:
        testEmbedding = img_to_encoding(frame[y : y + h, x : x + w], model, preprocess)
        min_dist = -1
        face_name = ""
        for face in list(database):
            for image in database[face]:
                similarityScore = similarity_score(testEmbedding, image)
                if similarityScore.item() > min_dist:
                    min_dist = similarityScore.item()
                    face_name = face
                if min_dist <= 0 :
                    face_name = "Unknown"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{face_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"{min_dist:.2f}", (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)


def main():
    model = mobilenet_v2()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('facenet_model.pth', map_location=device))
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    with open("./path_dict.p", "rb") as f:
        paths = pickle.load(f)

    with open("./database_dict.p", "rb") as f:
        database = pickle.load(f)

    # st.title("Real-time Face Recognition")

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faceRecognition(frame, database, model, preprocess)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    # st.title("Face Recognition App")
    # st.write("Upload an image for face recognition")
    
    # uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    # if uploaded_image is not None:
    #     image = Image.open(uploaded_image).convert("RGB")
    #     image = image.rotate(90, expand = True)
    #     st.image(image, caption="Uploaded Image", use_column_width=True)
    #     if st.button("Recognize"):
    #         result, image_cropped = faceRecognition(image, database, model, preprocess)
    #         st.write("Recognition Result:", result)
    #         st.image(image_cropped, caption = "cropped image", use_column_width=True)
    
if __name__ == "__main__":
    main()