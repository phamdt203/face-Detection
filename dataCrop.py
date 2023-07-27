import os
import cv2
from mtcnn import MTCNN

lfw_dataset_dir = r'C:\Users\Dell\OneDrive\Desktop\Code\AI\AI\faceRecognition\lfw'
output_dir = r'C:\Users\Dell\OneDrive\Desktop\Code\AI\AI\faceRecognition\cropped'

face_detector = MTCNN()

def crop_faces_from_image(image_path, output_dir):
    image_name = os.path.basename(image_path)
    person_name = os.path.basename(os.path.dirname(image_path))
    output_person_dir = os.path.join(output_dir, person_name)
    os.makedirs(output_person_dir, exist_ok=True)
    image = cv2.imread(image_path)
    detections = face_detector.detect_faces(image)
    for i, face_rect in enumerate(detections):
        x, y, w, h = face_rect['box']
        left, top, right, bottom = x, y, x + w, y + h
        face = image[top:bottom, left:right]
        face_name = f"{person_name}_{i}.jpg"
        output_path = os.path.join(output_person_dir, face_name)
        cv2.imwrite(output_path, face)

image_files = []
for root, dirs, files in os.walk(lfw_dataset_dir):
    image_files.extend([os.path.join(root, file) for file in files if file.endswith(".jpg")])

for image_file in image_files:
    crop_faces_from_image(image_file, output_dir)
