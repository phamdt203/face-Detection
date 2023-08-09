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
from matplotlib import cm

def img_to_encoding(image_path, model, device, preprocess):
    image_tensor = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image_tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(image_tensor)

def similarity_score(firstEmbedding, secondEmbedding):
    return F.pairwise_distance(firstEmbedding, secondEmbedding)

def faceDetector(faceDetector, image_path):    
    image = cv2.imread(image_path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceRects = faceDetector.detectMultiScale(gray_img, 1.1, 9)
    face = np.zeros((image.shape[0], image.shape[1]))
    for (x, y, w, h) in faceRects:
        left, top, right, bottom = x, y, x + w, y + h
        face = image[top : bottom, left : right]
    return face, faceRects

def Recognized(testEmbedding, database):
    print(similarity_score(testEmbedding, database['Aaron_Eckhart'][0]).item())
    min_dist = 1000
    face_name = ""
    for face in database.keys():
        if len(database[face]) < 1:
            continue
        similarityScore = similarity_score(database[face][0], testEmbedding)
        similarityScore = nn.Sigmoid(similarityScore)
        if similarityScore.item() < min_dist:
            min_dist = similarityScore.item()
            face_name = face
    print(face_name, min_dist)
    return face_name

def faceRecognition(database, paths, device, model, preprocess):
    for face in list(database)[:5]:
        paths[face] = paths[face].replace('\\', '/')
        if os.path.exists(paths[face]):
            image_path = os.path.join(paths[face], os.listdir(paths[face])[0])
            image = cv2.imread(image_path)
            fd = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            image_cropped, faceRects = faceDetector(fd, os.path.join(paths[face].replace("cropped", "lfw"), os.listdir(paths[face].replace("cropped", "lfw"))[0]))
            os.makedirs(f"test_cropped/{face}", exist_ok=True)
            cv2.imwrite(f"test_cropped/{face}/{face}.jpg",image_cropped)
            print(face)
            testEmbedding = img_to_encoding(f"test_cropped/{face}/{face}.jpg", model, device, preprocess)
            face_name = Recognized(testEmbedding, database)
            image = np.array(image)
            for (x, y, w, h) in faceRects:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                font_scale = 0.75
                font_thickness = 2
                text_size = cv2.getTextSize(face_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                text_x = int(x + (w - text_size[0][0]) // 2)
                text_y = int(y - 10)
                cv2.putText(image, face_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
            cv2.imwrite(f"output/{face_name}.jpg", image)


model = mobilenet_v2()
model.load_state_dict(torch.load("facenet_model.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
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
    
faceRecognition(database, paths, device, model, preprocess)