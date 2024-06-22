from ultralytics import YOLO
import os
import cv2
import json

amount_of_iterations = 80000
#divide by 3
learning_rate = 0.0005

def train_yolo_model(data_path, output_dir):
    model = YOLO('yolov8n.pt')  # Initialize YOLO model with a pre-trained model

    
    
    #print("Training model")
    #print(data_path)
    #get folderpath which is datapath without .yaml
    folder_path = data_path.split(".")[0] + "/labels"
    amount_of_images = len(os.listdir(folder_path))

    model.train(
        data=data_path,
        epochs= (amount_of_iterations // amount_of_images) + 1,  # YOLO training uses epochs instead of iterations
        lr0=learning_rate,
        optimizer='Adam',
        imgsz=640,
        batch=2,  # Adjust based on your GPU memory
        #steps at which to decrease learning rate 60% and 80% of the max iterations
        #step=[int(0.6 * amount_of_iterations), int(0.8 * amount_of_iterations)],
        #gamma is the factor by which the learning rate is reduced
        #warmup of 1000 iterations
        #warmup=1000,
        project=output_dir,
        name='yolov8_training',
        exist_ok=True,
        workers=0
        
    )

    return os.path.join(output_dir, 'yolov8_training', 'weights', 'best.pt')

# def predict_yolo(image_path, model):
#     results = model(image_path)
#     if len(results) == 0:
#         return "Unknown"
#     if len(results[0].boxes.cls) == 0:
#         return "Unknown"
#     return results[0].names[results[0].boxes.cls[0].item()]


def predict_normal(image_path, model):
    results = model(image_path)
    if len(results) == 0:
        return "Unknown"
    if len(results[0].boxes.cls) == 0:
        return "Unknown"
    
        
    return results[0].names[results[0].boxes.cls[0].item()]

def train_hierarchical_models():


    if os.path.exists("output/normal/yolov8_training/weights/best.pt"):
        normal_weights = "output/normal/yolov8_training/weights/best.pt"
    else:
        normal_weights = train_yolo_model("data/result/yolo_normal.yaml", "output/normal")


    if os.path.exists("output/root/yolov8_training/weights/best.pt"):
        root_weights = "output/root/yolov8_training/weights/best.pt"
    else:
        root_weights = train_yolo_model("data/result/yolo_root.yaml", "output/root")


    if os.path.exists("output/chair_sofa/yolov8_training/weights/best.pt"):
        chair_sofa_weights = "output/chair_sofo/yolov8_training/weights/best.pt"
    else:
        chair_sofa_weights = train_yolo_model( "data/result/yolo_chair_sofa.yaml", "output/chair_sofa")

    if os.path.exists("output/cat_dog/yolov8_training/weights/best.pt"):
        cat_dog_weights = "output/cat_dog/yolov8_training/weights/best.pt"
    else:
        cat_dog_weights = train_yolo_model( "data/result/yolo_cat_dog.yaml", "output/cat_dog")

    if os.path.exists("output/cow_sheep/yolov8_training/weights/best.pt"):
        cow_sheep_weights = "output/cow_sheep/yolov8_training/weights/best.pt"
    else:
        cow_sheep_weights = train_yolo_model( "data/result/yolo_cow_sheep.yaml", "output/cow_sheep")

    if os.path.exists("output/car_bus/yolov8_training/weights/best.pt"):
        bus_car_weights = "output/car_bus/yolov8_training/weights/best.pt"
    else:
        bus_car_weights = train_yolo_model( "data/result/yolo_car_bus.yaml", "output/car_bus")

    if os.path.exists("output/motorbike_bicycle/yolov8_training/weights/best.pt"):
        bicycle_motorbike_weights = "output/motorbike_bicycle/yolov8_training/weights/best.pt"
    else:
        bicycle_motorbike_weights = train_yolo_model( "data/result/yolo_motorbike_bicycle.yaml", "output/motorbike_bicycle")

    return {
        "normal": normal_weights,
        "root": root_weights,
        #"cat_dog_cow_sheep": cat_dog_cow_sheep_weights,
        #"motorbike_bicycle_car_bus": motorbike_bicycle_car_bus_weights,
        "chair_sofa": chair_sofa_weights,
        "cat_dog": cat_dog_weights,
        "cow_sheep": cow_sheep_weights,
        "bus_car": bus_car_weights,
        "bicycle_motorbike": bicycle_motorbike_weights
    }

def predict_hierarchy(image_path, models):
    def predict_node(image_path, model):
        results = model(image_path)
        if len(results) == 0:
            return "Unknown"

        if len(results[0].boxes.cls) == 0:
            return "Unknown"
        


        
        return results[0].names[results[0].boxes.cls[0].item()]

        

    root_prediction = predict_node(image_path, models["root"])

    #print("Root prediction: " + root_prediction)

    if root_prediction == "cat_dog":
        return predict_node(image_path, models["cat_dog"])
    elif root_prediction == "cow_sheep":
        return predict_node(image_path, models["cow_sheep"])
    elif root_prediction == "car_bus":
        return predict_node(image_path, models["bus_car"])
    elif root_prediction == "motorbike_bicycle":
        return predict_node(image_path, models["bicycle_motorbike"])
    elif root_prediction == "chair_sofa":
        return predict_node(image_path, models["chair_sofa"])
    else:
        return "Unknown"

    # if root_prediction == "cat_dog_cow_sheep":
    #     cat_dog_cow_sheep_prediction = predict_node(image_path, models["cat_dog_cow_sheep"])
    #     if cat_dog_cow_sheep_prediction == "cat_dog":
    #         return predict_node(image_path, models["cat_dog"])
    #     elif cat_dog_cow_sheep_prediction == "cow_sheep":
    #         return predict_node(image_path, models["cow_sheep"])
    # elif root_prediction == "motorbike_bicycle_car_bus":
    #     motorbike_bicycle_car_bus_prediction = predict_node(image_path, models["motorbike_bicycle_car_bus"])
    #     if motorbike_bicycle_car_bus_prediction == "motorbike_bicycle":
    #         return predict_node(image_path, models["bicycle_motorbike"])
    #     elif motorbike_bicycle_car_bus_prediction == "car_bus":
    #         return predict_node(image_path, models["bus_car"])
    # elif root_prediction == "sofa_chair":
    #     return predict_node(image_path, models["chair_sofa"])
    
    

def load_models(): 
    return {
        # "normal": YOLO("output/normal/best.pt"),    
        # "root": YOLO("output/root/yolov8_training/weights/best.pt"),
        # #"cat_dog_cow_sheep": YOLO("output/cat_dog_cow_sheep/best.pt"),
        # #"motorbike_bicycle_car_bus": YOLO("output/motorbike_bicycle_car_bus/best.pt"),
        # "chair_sofa": YOLO("output/chair_sofa/best.pt"),
        # "cat_dog": YOLO("output/cat_dog/best.pt"),
        # "cow_sheep": YOLO("output/cow_sheep/best.pt"),
        # "bus_car": YOLO("output/bus_car/best.pt"),
        # "bicycle_motorbike": YOLO("output/bicycle_motorbike/best.pt")
        "normal": YOLO("output/normal/yolov8_training/weights/best.pt"),
        "root": YOLO("output/root/yolov8_training/weights/best.pt"),
        "chair_sofa": YOLO("output/chair_sofa/yolov8_training/weights/best.pt"),
        "cat_dog": YOLO("output/cat_dog/yolov8_training/weights/best.pt"),
        "cow_sheep": YOLO("output/cow_sheep/yolov8_training/weights/best.pt"),
        "bus_car": YOLO("output/car_bus/yolov8_training/weights/best.pt"),
        "bicycle_motorbike": YOLO("output/motorbike_bicycle/yolov8_training/weights/best.pt")

    }

def test_accuracy():
    weights = load_models()

    total = 0
    correct_hierarchy = 0
    correct_normal = 0

    validation_annotations_file = 'data/result/single_label/annotations.json'
    validation_images_folder = 'data/result/single_label/images'
    
    with open(validation_annotations_file, 'r') as f:
        validation_data = json.load(f)

    image_id_to_file_name = {img['id']: img['file_name'] for img in validation_data['images']}

    for annotation in validation_data['annotations']:
        image_id = annotation['image_id']
        file_name = image_id_to_file_name[image_id]
        image_path = os.path.join(validation_images_folder, file_name)
        
        total += 1
        class_name = next(category['name'] for category in validation_data['categories'] if category['id'] == annotation['category_id'])


        with open("output/log.txt", "a") as log:
            log.write("Actual class: " + class_name + " ")
        

        prediction_hierarchy = predict_hierarchy(image_path, weights)

        with open("output/log.txt", "a") as log:
            log.write("Hierarchical model prediction: " + prediction_hierarchy + "\n")
        prediction_normal = predict_normal(image_path, weights["normal"])

        with open("output/log.txt", "a") as log:
            log.write("Actual class: " + class_name + " ")
        with open("output/log.txt", "a") as log:
            log.write("Normal model prediction: " + prediction_normal + "\n")
        
        if prediction_hierarchy == class_name:
            correct_hierarchy += 1
        
        if prediction_normal == class_name:
            correct_normal += 1

        print("actual class: " + class_name + " hierarchical model prediction: " + prediction_hierarchy + " normal model prediction: " + prediction_normal)

        if total % 100 == 0:
            print(f'Processed {total} images')
            print(f'Hierarchical model accuracy: {correct_hierarchy / total}')
            print(f'Normal model accuracy: {correct_normal / total}')

    print(f'Hierarchical model accuracy: {correct_hierarchy / total}')
    print(f'Normal model accuracy: {correct_normal / total}')

if __name__ == '__main__':
    train_hierarchical_models()
    test_accuracy()
