import os
import cv2
import pickle

lfw_dataset_dir = r'lfw'
output_dir = r'cropped'

def crop_faces_from_image( image_path, output_dir = None):
    image_name = os.path.basename(image_path)
    person_name = os.path.basename(os.path.dirname(image_path))
    output_person_dir = os.path.join(output_dir, person_name)
    os.makedirs(output_person_dir, exist_ok=True)
    image = cv2.imread(image_path)
    print(image_path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)
    for (x, y, w, h) in faces_rect:
        left, top, right, bottom = x, y, x + w, y + h
        face = image[top : bottom, left : right]
        face_name = f"{image_name}"
        output_path = os.path.join(output_person_dir, face_name)
        cv2.imwrite(output_path, face)
    return person_name, output_person_dir

image_files = []
for root, dirs, files in os.walk(lfw_dataset_dir):
    image_files.extend([os.path.join(root, file_image) for file_image in files if file_image.endswith(".JPG")])

path_dict = {}

for index, image_file in enumerate(image_files):
    person_name, output_person_dir = crop_faces_from_image(image_file, output_dir)
    path_dict[person_name] = output_person_dir

with open("path_dict.p", "wb") as f:
    pickle.dump(path_dict, f)