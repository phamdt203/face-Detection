import torch
import pickle
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from data import *
from parameters import *
import cv2
import numpy as np
from loss import *
from faceDetector import *
import os

def load_path():
    with open("./path_dict.p", "rb") as f:
        paths = pickle.load(f)
    return paths

def load_face(paths):
    faces = []
    for key in paths.keys():
        paths[key] = paths[key].replace("\\", "/")
        faces.append(key)
    return faces

def load_images(paths, transform):
    images = {}
    for key in paths.keys():
        li = []
        for img in os.listdir(paths[key]):
            image = cv2.imread(os.path.join(paths[key],img))
            image = transform(image)
            li.append(image)
        images[key] = np.array(li)
    return images

def img_to_encoding(image_path, model, transform):
    image = cv2.imread(image_path)
    image = transform(image)
    embedding = model(image)
    return embedding

def verify(image_path, identity, database, model):
    encoding = img_to_encoding(image_path, model, False)
    min_dist = 1000
    for  pic in database:
        dist = np.linalg.norm(encoding - pic)
        if dist < min_dist:
            min_dist = dist
    print(identity + ' : ' +str(min_dist)+ ' ' + str(len(database)))
    
    if min_dist<THRESHOLD:
        door_open = True
    else:
        door_open = False
        
    return min_dist, door_open

def load_database(faces, paths, model):
    database = {}
    for face in faces:
        database[face] = []

    for face in faces:
        for img in os.listdir(paths[face]):
            database[face].append(img_to_encoding(os.path.join(paths[face],img), model))
    return database

def faceRecognition(faceDetector, imgae_path,database, faces, model):
    image = cv2.read(imgae_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceRects = faceDetector.detect(gray)
    image_path = "test_image"
    for (x, y, w, h) in faceRects:
        roi = image[y:y+h,x:x+w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi,(IMAGE_SIZE, IMAGE_SIZE))
        min_dist = 1000
        identity = ""
        detected  = False
        for face in range(len(faces)):
            person = faces[face]
            dist, detected = verify(roi, person, database[person], model)
            if detected and dist < min_dist:
                min_dist = dist
                identity = person
        image_path = os.path.join(image_path, f"{identity}.jpg")
        if detected:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, identity, (x+ (w//2),y-2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_AA)
    cv2.imread(image_path, image)

def test_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    model = model.to(device)
    model.load_state_dict(torch.load("facenet_model.pth"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    fd = faceDetector('haarcascade_frontalface_default.xml')
    paths = load_path()
    print(paths)
    faces = load_face(paths)
    images = load_images(paths, transform)
    database = load_database(faces, paths, model)
    faceRecognition()
