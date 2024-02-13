from ultralytics import YOLO

from matplotlib.pyplot import figure
import matplotlib.image as image
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from numpy import asarray
from PIL import Image
import os

model_path = 'weights/yolo.pt'
images_folder = 'images'
private_folder = 'private'

model_trained = YOLO(model_path)

image_files = [os.path.join(images_folder, file) for file in os.listdir(images_folder) if file.endswith(".jpg")]

for image_file in image_files:
    results = model_trained.predict(source=image_file, line_thickness=1, conf=0.2, augment=False, save_txt=True,
                                    save=True)

    # Получаем имя файла без пути и расширения
    image_filename = os.path.basename(image_file)
    image_name = os.path.splitext(image_filename)[0]

    private_images_folder = os.path.join(private_folder, 'images')
    private_labels_folder = os.path.join(private_folder, 'labels')

    # Создаем папку для сохранения картинок, если она не существует
    os.makedirs(private_images_folder, exist_ok=True)
    os.makedirs(private_labels_folder, exist_ok=True)

    # Перемещаем предсказанные файлы с картинками
    os.rename(f"runs/detect/predict/{image_name}.jpg", os.path.join(private_images_folder, f"{image_name}.jpg"))

    # Перемещаем предсказанные текстовые файлы
    os.rename(f"runs/detect/predict/labels/{image_name}.txt", os.path.join(private_labels_folder, f"{image_name}.txt"))

os.system("rmdir /S /Q runs")
