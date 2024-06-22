from detectron2.data.datasets import register_coco_instances
import os
import torch, detectron2
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import json
import numpy as np

import os
import logging

# Suppress Detectron2 logging
logger = logging.getLogger("detectron2")
logger.setLevel(logging.WARNING)

# Suppress TensorFlow logging if necessary
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


amount_of_iterations = 25000

learning_rate = 0.0005


def predict_normal(image_path, predictor, cfg):
    image = cv2.imread(image_path)
    outputs = predictor(image)
    classes = outputs["instances"].pred_classes.to("cpu").numpy()
    if len(classes) == 0:
        return "Unknown"
    
    max_class_index = len(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes) - 1
    valid_classes = [i for i in classes if i <= max_class_index]

    if not valid_classes:

        return ["Unknown"]
    return [MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[i] for i in valid_classes][0]

def train_hierarchical_models():
    def train_model(classes, output_dir, name):



        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))

        
        cfg.DATASETS.TRAIN = (name + "_train",)
        cfg.DATASETS.TEST = (name + "_val",)
        cfg.DATALOADER.NUM_WORKERS = 4  
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Use COCO pretrained weights
        cfg.SOLVER.IMS_PER_BATCH = 2  
        cfg.SOLVER.BASE_LR = learning_rate
        cfg.SOLVER.MAX_ITER = amount_of_iterations
        #steps at which to decrease learning rate 60% and 80% of the max iterations
        cfg.SOLVER.STEPS = (int(0.6 * amount_of_iterations), int(0.8 * amount_of_iterations))
        cfg.SOLVER.WARMUP_ITERS = 1000
        cfg.SOLVER.WARMUP_METHOD = "linear"
        cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
        cfg.SOLVER.GAMMA = 0.1
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
        
        cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
        cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
        cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0  

        os.makedirs(output_dir, exist_ok=True)
        cfg.OUTPUT_DIR = output_dir

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        return os.path.join(output_dir, "model_final.pth")


    #don't train if already trained
    if os.path.exists("output/normal/model_final.pth"):
        normal_weights = "output/normal/model_final.pth"
    else:
        print(MetadataCatalog.list())
        register_coco_instances("normal_train", {}, "data/result/normalcar/annotations.json", "data/result/normalcar/images")
        register_coco_instances("normal_val", {}, "data/result/normalcar/annotations.json", "data/result/normalcar/images")
        MetadataCatalog.get("normal_train").thing_classes = ["trafficLight-Green", "trafficLight-Yellow", "biker", "pedestrian", "trafficLight-Red"]
        MetadataCatalog.get("normal_val").thing_classes = ["trafficLight-Green", "trafficLight-Yellow", "biker", "pedestrian", "trafficLight-Red"]

        normal_weights = train_model(["trafficLight-Green", "trafficLight-Yellow", "biker", "pedestrian", "trafficLight-Red"], "output/normal", "normal")

    if os.path.exists("output/root/model_final.pth"):
        root_weights = "output/root/model_final.pth"
    else:

        register_coco_instances("root_train", {}, "data/result/groupedcar/annotations.json", "data/result/groupedcar/images")
        register_coco_instances("root_val", {}, "data/result/groupedcar/annotations.json", "data/result/groupedcar/images")

        MetadataCatalog.get("root_train").thing_classes = ["trafficLight-Green_trafficLight-Yellow", "biker_pedestrian_trafficLight-Red"]
        MetadataCatalog.get("root_val").thing_classes = ["trafficLight-Green_trafficLight-Yellow", "biker_pedestrian_trafficLight-Red"]

        root_weights = train_model(["trafficLight-Green_trafficLight-Yellow", "biker_pedestrian_trafficLight-Red"], "output/root", "root")




    return {
        "normal": normal_weights,
        "root": root_weights,
    }

def predict_hierarchy(image_path, cfgs, predictors):
    def predict_node(image, cfg, predictor):
        outputs = predictor(image)
        classes = outputs["instances"].pred_classes.to("cpu").numpy()
        #print("classes", classes)
        # visualizer = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        # out_image = out.get_image()[:, :, ::-1]
        # cv2.imshow("Prediction", out_image)
        # cv2.waitKey(0)
        #print("classes", classes)
        #print("metadata ", MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)
        if len(classes) == 0:
            return "Unknown"
        max_class_index = len(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes) - 1
        valid_classes = [i for i in classes if i <= max_class_index]
        #print("valid_classes", valid_classes)
        if not valid_classes:
            print("Warning: Detected class indices exceed the number of known classes.")
            return ["Unknown"]
        return [MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[i] for i in valid_classes][0]

    image = cv2.imread(image_path)

    #write to logs what each node is predicting


    root_prediction = predict_node(image, cfgs["root"], predictors["root"])

    print("root_prediction", root_prediction)

    if root_prediction == "trafficLight-Green_trafficLight-Yellow":
        with open("output/log.txt", "a") as log:
            log.write(f"Root prediction: {root_prediction} and then ")
        return "trafficLight-Green_trafficLight-Yellow"

    elif root_prediction == "biker_pedestrian_trafficLight-Red":
        with open("output/log.txt", "a") as log:
            log.write(f"Root prediction: {root_prediction} and then ")
        return "biker_pedestrian_trafficLight-Red"
    return "Unknown"
    



def load_weights():
    return {
        "normal": "output/normal/model_final.pth",	
        "root": "output/root/model_final.pth",
    }

def register_and_set_classes():


    if "normal_train" in DatasetCatalog.list():
        return
    else:
        register_coco_instances("normal_train", {}, "data/result/normalcar/annotations.json", "data/result/normalcar/images")
        register_coco_instances("normal_val", {}, "data/result/normalcar/annotations.json", "data/result/normalcar/images")
        MetadataCatalog.get("normal_train").thing_classes = ["trafficLight-Green", "trafficLight-Yellow", "biker", "pedestrian", "trafficLight-Red"]
        MetadataCatalog.get("normal_val").thing_classes = ["trafficLight-Green", "trafficLight-Yellow", "biker", "pedestrian", "trafficLight-Red"]

    if "root_train" in DatasetCatalog.list():
        return
    else:
        register_coco_instances("root_train", {}, "data/result/groupedcar/annotations.json", "data/result/groupedcar/images")
        register_coco_instances("root_val", {}, "data/result/groupedcar/annotations.json", "data/result/groupedcar/images")
        MetadataCatalog.get("root_train").thing_classes = ["trafficLight-Green_trafficLight-Yellow", "biker_pedestrian_trafficLight-Red"]
        MetadataCatalog.get("root_val").thing_classes = ["trafficLight-Green_trafficLight-Yellow", "biker_pedestrian_trafficLight-Red"]
    
   



def test_accuracy():
    register_and_set_classes()
    print(MetadataCatalog.list())
    weights = load_weights()


    cfgs = {}
    predictors = {}
    
    for key, weight in weights.items():
        print("key: ", key)
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = weight
        
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(f"{key}_train").thing_classes)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
        print("length of classes: ", len(MetadataCatalog.get(f"{key}_train").thing_classes))
        cfg.DATASETS.TRAIN = (f"{key}_train",)
        cfg.DATASETS.TEST = (f"{key}_val",)
        cfg.DATALOADER.NUM_WORKERS = 2
        cfgs[key] = cfg
        predictors[key] = DefaultPredictor(cfg)

    total = 0
    correct_hierarchy = 0
    correct_normal = 0

    # Load validation data
    validation_annotations_file = 'data/result/testcar/annotations.json'
    validation_images_folder = 'data/result/testcar/images'
    
    with open(validation_annotations_file, 'r') as f:
        validation_data = json.load(f)

    # Create a mapping from image id to file name
    image_id_to_file_name = {img['id']: img['file_name'] for img in validation_data['images']}

    for annotation in validation_data['annotations']:
        image_id = annotation['image_id']
        file_name = image_id_to_file_name[image_id]
        image_path = os.path.join(validation_images_folder, file_name)
        
        total += 1
        class_name = next(category['name'] for category in validation_data['categories'] if category['id'] == annotation['category_id'])
        with open("output/log.txt", "a") as log:
            log.write(f"Actual: {class_name} and ")
        prediction_hierarchy = predict_hierarchy(image_path, cfgs, predictors)
        prediction_normal = predict_normal(image_path, predictors["normal"], cfgs["normal"])
        

        with open("output/log.txt", "a") as log:
            log.write(prediction_hierarchy + "\n")
            log.write(f'Actual: {class_name} and Normal Prediction: {prediction_normal}\n')



        if prediction_hierarchy == class_name:
            correct_hierarchy += 1
        
        if prediction_normal == class_name:
            correct_normal += 1

    print(f'Hierarchical model accuracy: {correct_hierarchy / total}')
    print(f'Normal model accuracy: {correct_normal / total}')

if __name__ == '__main__':
    train_hierarchical_models()
    test_accuracy()


