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
    new_images = []
    new_annotations = []
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
    image_id_mapping = {}
    for image in coco_data['images']:
        image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image['id']]
        filtered_annotations = []
        for annotation in image_annotations:
            category_name = next(category['name'] for category in coco_data['categories'] if category['id'] == annotation['category_id'])
            if category_name in class_mapping:
                new_category_name = class_mapping[category_name]
                new_category_id = category_mapping[new_category_name]
                annotation['category_id'] = new_category_id
                filtered_annotations.append(annotation)

        if filtered_annotations:
            new_image_id = len(new_images) + 1
            image_id_mapping[image['id']] = new_image_id
            new_images.append({
                'id': new_image_id,
                'file_name': image['file_name'],
                'height': image['height'],
                'width': image['width']
            })
            for annotation in filtered_annotations:
                annotation['image_id'] = new_image_id
                new_annotations.append(annotation)
            # Copy image to destination folder
            shutil.copy(os.path.join(source_folder, image['file_name']), os.path.join(destination_images_folder, image['file_name']))

    # Create new annotation file
    new_coco_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': new_categories
    }
    with open(os.path.join(destination_folder, 'annotations.json'), 'w') as f:
        json.dump(new_coco_data, f, indent=4)


classes_to_keep = ["trafficLight-Green", "trafficLight-Red", "trafficLight-Yellow", "biker", "pedestrian"]
#class_groups = [["trafficLight-Green", "trafficLight-Yellow"], ["biker", "pedestrian", "trafficLight-Red"]]
class_groups = [["trafficLight-Green"], ["trafficLight-Yellow"], ["biker"], ["pedestrian"], ["trafficLight-Red"]]
source_folder = 'cardata/test'
destination_folder = 'data/result/train'
annotations_file = 'cardata/test/annotations.json'

filter_or_combine_classes(classes_to_keep, class_groups, source_folder, destination_folder, annotations_file)


#should make a subset of the dataset such that it only has images which only has 1 class in it of cat, dog, cow, sheep, motorbike, bicycle, car, bus, sofa, chair
def filter_out_images_with_multiple_labels(annotations_file, classes_to_keep, destination_folder):
    # Load the COCO annotations
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)

    # Create the destination directories
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    destination_images_folder = os.path.join(destination_folder, "images")
    if not os.path.exists(destination_images_folder):
        os.makedirs(destination_images_folder)

    # Filter images with multiple labels
    new_images = []
    new_annotations = []
    new_categories = []
    category_mapping = {}
    for category_name in classes_to_keep:
        new_category_id = len(new_categories) + 1
        category_mapping[category_name] = new_category_id
        new_categories.append({
            'id': new_category_id,
            'name': category_name,
            'supercategory': 'none'
        })

    image_id_mapping = {}
    for image in coco_data['images']:
        image_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image['id']]
        if len(image_annotations) == 1:
            annotation = image_annotations[0]
            category_name = next(category['name'] for category in coco_data['categories'] if category['id'] == annotation['category_id'])
            if category_name in classes_to_keep:
                new_image_id = len(new_images) + 1
                image_id_mapping[image['id']] = new_image_id
                new_images.append({
                    'id': new_image_id,
                    'file_name': image['file_name'],
                    'height': image['height'],
                    'width': image['width']
                })
                annotation['image_id'] = new_image_id
                new_annotations.append(annotation)
                # Copy image to destination folder
                shutil.copy(os.path.join(source_folder, image['file_name']), os.path.join(destination_images_folder, image['file_name']))

    # Create new annotation file
    new_coco_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': new_categories
    }
    with open(os.path.join(destination_folder, 'annotations.json'), 'w') as f:
        json.dump(new_coco_data, f, indent=4)


        