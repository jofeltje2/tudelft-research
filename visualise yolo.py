import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

path_to_images = 'data/result/yolo_normal/images'
path_to_annotations = 'data/result/yolo_normal/labels'

def visualize_images(path_to_images, path_to_annotations):
    for image_name in os.listdir(path_to_images):
        img = cv2.imread(os.path.join(path_to_images, image_name))
        h, w, _ = img.shape
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        with open(os.path.join(path_to_annotations, image_name.replace('.jpg', '.txt')), 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.split())
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='r', linewidth=2))
                plt.text(x1, y1, str(int(class_id)), fontsize=12, color='r')
        plt.show()


visualize_images(path_to_images, path_to_annotations)