import json
import os
import shutil

def filter_or_combine_classes(classes_to_keep, class_groups, source_folder, destination_folder, annotations_file):
    # Load the COCO annotations
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    # Create a mapping from class names to new class groups
    class_mapping = {class_name: class_name for class_name in classes_to_keep}
    for group in class_groups:
        group_name = "_".join(group)
        for class_name in group:
            class_mapping[class_name] = group_name

    # Create the destination directories
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    destination_images_folder = os.path.join(destination_folder, "images")
    if not os.path.exists(destination_images_folder):
        os.makedirs(destination_images_folder)

    # Filter and combine classes in annotations
    new_categories = []
    category_mapping = {}

    # Ensuring the categories are in the same order as class_groups
    ordered_classes = [class_name for group in class_groups for class_name in group]
    ordered_classes += [class_name for class_name in classes_to_keep if class_name not in ordered_classes]
    ordered_class_groups = sorted(set(class_mapping.values()), key=lambda x: ordered_classes.index(x.split("_")[0]))

    for category_name in ordered_class_groups:
        new_category_id = len(new_categories) + 1
        category_mapping[category_name] = new_category_id
        new_categories.append({
            'id': new_category_id,
            'name': category_name,
            'supercategory': 'none'
        })

    # Copy and update annotations
    for image in coco_data['images']:
        image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image['id']]
        filtered_annotations = []
        for annotation in image_annotations:
            category_name = next(category['name'] for category in coco_data['categories'] if category['id'] == annotation['category_id'])
            if category_name in class_mapping:
                new_category_name = class_mapping[category_name]
                new_category_id = category_mapping[new_category_name] - 1  # YOLO class indices start at 0
                bbox = annotation['bbox']
                x_center = (bbox[0] + bbox[2] / 2) / image['width']
                y_center = (bbox[1] + bbox[3] / 2) / image['height']
                w = bbox[2] / image['width']
                h = bbox[3] / image['height']
                filtered_annotations.append(f"{new_category_id} {x_center} {y_center} {w} {h}")

        if filtered_annotations:
            # Save annotations in YOLO format
            txt_file_path = os.path.join(destination_folder, os.path.splitext(image['file_name'])[0] + '.txt')
            with open(txt_file_path, 'w') as txt_file:
                txt_file.write("\n".join(filtered_annotations))

            # Copy image to destination folder
            shutil.copy(os.path.join(source_folder, image['file_name']), os.path.join(destination_images_folder, image['file_name']))



