import pickle
import os
from PIL import Image
import torch

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
        if os.path.exists(paths[key]):
          for img in os.listdir(paths[key]):
            image = Image.open(os.path.join(paths[key],img)).convert("RGB")
            image = transform(image)
            li.append(image)
        images[key] = li
    return images

def img_to_encoding(image_path, model, transform, device):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(image_tensor)

def load_database(model, transform, device):
    paths = load_path()
    faces = load_face(paths)
    database = {}
    for face in faces:
        database[face] = []

    for face in faces:
      if os.path.exists(paths[face]):
        for img in os.listdir(paths[face]):
            database[face].append(img_to_encoding(os.path.join(paths[face],img), model, transform, device))
    return database