from typing import Union
from PIL import Image, ImageDraw
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File
import io
from facenet_pytorch import InceptionResnetV1
import torch
import cv2
import os
import pickle
import sys

app = FastAPI()

def faceRecognition(image_path):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    recognized_image = Image.open(image_path)

    model = InceptionResnetV1()
    model = model.to(device)
    model.load_state_dict(torch.load("facenet_model.pth"))

    with open("./path_dict.p", 'rb') as f:
        paths = pickle.load(f)

    faces = []
    for key in paths.keys():
        faces.append(key)
    
    if(len(faces) == 0):
        print("No images found in database!!")
        print("Please add images to database")
        sys.exit()

    image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)
    for (x, y, w, h) in faces_rect:
        left, top, right, bottom = x, y, x + w, y + h
        draw.rectangle([left, top], [right, bottom], outline = "blue", width= 4)
        draw.text((x + 5, y + 5), person_name, fill = "blue")
    
    return recognized_image

@app.post("/face_recognition/")
async def perform_face_recognition(image: UploadFile = File(...)):
    # Đọc ảnh từ dữ liệu gửi lên
    contents = await image.read()
    img = Image.open(io.BytesIO(contents))

    # Lưu ảnh đã xử lý vào thư mục tạm và trả về đường dẫn đến ảnh
    temp_image_path = "processed_image.jpg"
    img.save(temp_image_path)
    recognized_image = faceRecognition(temp_image_path)

    # Trả về ảnh đã xử lý
    return FileResponse(recognized_image, media_type="image/jpeg")