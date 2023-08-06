import cv2
import torch
import pickle
from torchvision import transforms
from PIL import Image
from parameters import *
from faceDetector import *
import torch.nn.functional as F
from model import mobilenet_v2

def img_to_encoding(image_tensor, model, device, preprocess):
    image_tensor = preprocess(image_tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(image_tensor)

def similarity_score(firstEmbedding, secondEmbedding):
    return F.cosine_similarity(firstEmbedding, secondEmbedding)

def faceDetector(faceDetector, image_path):    
    image = cv2.imread(image_path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceRects = faceDetector.detect(gray_img)
    for (x, y, w, h) in faceRects:
        left, top, right, bottom = x, y, x + w, y + h
        face = image[top : bottom, left : right]
    return torch.from_numpy(face), faceRects

def Recognized(testEmbedding, database):
    min_dist = 1000
    face_name = ""
    for face in database.keys():
        similarityScore = similarity_score(database[face], testEmbedding)
        if similarityScore < min_dist:
            min_dist = similarityScore
            face_name = face
    return face_name

def faceRecognition(database, paths, device, model, preprocess):
    for face in list(database)[:5]:
        if os.path.exists(paths[face]):
            image = Image.open(os.listdir(paths[face].replace('\\', '/'))[0]).convert("RGB")
            fd = faceDetector('haarcascade_frontalface_default.xml')
            image_tensor, faceRects = faceDetector(fd, paths[face])
            testEmbedding = img_to_encoding(image_tensor, model, device, preprocess)
            face_name = Recognized(testEmbedding, database)
            for (x, y, w, h) in faceRects:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                font_scale = 0.75
                font_thickness = 2
                text_size = cv2.getTextSize(face_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                text_x = x + (w - text_size[0]) // 2
                text_y = y - 10
                cv2.putText(image, face_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
            cv2.imwrite(f"output/{face_name}.jpg", image)

model = mobilenet_v2()
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