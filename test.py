import torch
import pickle
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from data import *
from parameters import *
from facenet_pytorch import InceptionResnetV1
import cv2
import numpy as np

model = InceptionResnetV1()
model.load_state_dict(torch.load("facenet_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

with open("./path_dict.p", "rb") as f:
    paths = pickle.load(f)

faces = []
for key in paths.keys():
    paths[key] = paths[key].replace("\\", "/")
    faces.append(key)

images = {}
for key in paths.keys():
    li = []
    for img in os.listdir(paths[key]):
        img1 = cv2.imread(os.path.join(paths[key],img))
        img2 = img1[...,::-1]
        li.append(np.around(np.transpose(img2, (2,0,1))/255.0, decimals=12))
    images[key] = np.array(li)
