import json
import os
import cv2
import matplotlib.pyplot as plt


path_to_images = 'data/result/motorbike_bicycle/images'
path_to_json = 'data/result/motorbike_bicycle/annotations.json'

def visualize_images(path_to_images, path_to_json):
    with open(path_to_json, 'r') as f:
        coco_data = json.load(f)

    # Extract class names
    categories = {category['id']: category['name'] for category in coco_data['categories']}

    for image in coco_data['images']:
        img = cv2.imread(os.path.join(path_to_images, image['file_name']))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image['id']:
                bbox = annotation['bbox']
                class_id = annotation['category_id']
                class_name = categories[class_id]
                plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='r', linewidth=2))
                plt.text(bbox[0], bbox[1], class_name, fontsize=12, color='r')
        plt.show()

visualize_images(path_to_images, path_to_json)

    
    
