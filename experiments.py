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


amount_of_iterations = 5000

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
        #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml"))
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/Faster_RCNN_R_50_FPN_1x.yaml"))
        cfg.DATASETS.TRAIN = (name + "_train",)
        cfg.DATASETS.TEST = (name + "_val",)
        cfg.DATALOADER.NUM_WORKERS = 4  
        #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_1x.yaml")
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/Faster_RCNN_R_50_FPN_1x.yaml")
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
        register_coco_instances("normal_train", {}, "data/result/normal/annotations.json", "data/result/normal/images")
        register_coco_instances("normal_val", {}, "data/result/normal/annotations.json", "data/result/normal/images")
        MetadataCatalog.get("normal_train").thing_classes = ["cat", "dog", "cow", "sheep", "motorbike", "bicycle", "car", "bus", "sofa", "chair"]
        MetadataCatalog.get("normal_val").thing_classes = ["cat", "dog", "cow", "sheep", "motorbike", "bicycle", "car", "bus", "sofa", "chair"]

        normal_weights = train_model(["cat", "cow", "dog", "sheep", "bicycle", "motorbike", "bus", "car", "sofa", "chair"], "output/normal", "normal")

    if os.path.exists("output/root/model_final.pth"):
        root_weights = "output/root/model_final.pth"
    else:

        # Train root model (distinguish between "cat_dog_cow_sheep", "motorbike_bicycle_car_bus", "chair_sofa")

        register_coco_instances("root_train", {}, "data/result/root/annotations.json", "data/result/root/images")
        register_coco_instances("root_val", {}, "data/result/root/annotations.json", "data/result/root/images")

        MetadataCatalog.get("root_train").thing_classes = ["cat_dog", "cow_sheep", "motorbike_bicycle", "car_bus", "sofa_chair"]
        MetadataCatalog.get("root_val").thing_classes = ["cat_dog", "cow_sheep", "motorbike_bicycle", "car_bus", "sofa_chair"]

        root_weights = train_model(["cat_dog", "cow_sheep", "motorbike_bicycle", "car_bus", "sofa_chair"], "output/root", "root")


    # if os.path.exists("output/cat_dog_cow_sheep/model_final.pth"):
    #     cat_dog_cow_sheep_weights = "output/cat_dog_cow_sheep/model_final.pth"
    # else:
    #     # Node for "cat_dog" vs "cow_sheep"

    #     register_coco_instances("cat_dog_cow_sheep_train", {}, "data/result/cat_dog_cow_sheep/annotations.json", "data/result/cat_dog_cow_sheep/images")
    #     register_coco_instances("cat_dog_cow_sheep_val", {}, "data/result/cat_dog_cow_sheep/annotations.json", "data/result/cat_dog_cow_sheep/images")

    #     MetadataCatalog.get("cat_dog_cow_sheep_train").thing_classes = ["cat_dog", "cow_sheep"]
    #     MetadataCatalog.get("cat_dog_cow_sheep_val").thing_classes = ["cat_dog", "cow_sheep"]

    #     cat_dog_cow_sheep_weights = train_model(["cat_dog", "cow_sheep"], "output/cat_dog_cow_sheep", "cat_dog_cow_sheep")

    # if os.path.exists("output/motorbike_bicycle_car_bus/model_final.pth"):
    #     motorbike_bicycle_car_bus_weights = "output/motorbike_bicycle_car_bus/model_final.pth"
    # else:

    #     # Node for "motorbike_bicycle" vs "car_bus"

    #     register_coco_instances("motorbike_bicycle_car_bus_train", {}, "data/result/motorbike_bicycle_car_bus/annotations.json", "data/result/motorbike_bicycle_car_bus/images")
    #     register_coco_instances("motorbike_bicycle_car_bus_val", {}, "data/result/motorbike_bicycle_car_bus/annotations.json", "data/result/motorbike_bicycle_car_bus/images")

    #     MetadataCatalog.get("motorbike_bicycle_car_bus_train").thing_classes = ["motorbike_bicycle", "car_bus"]
    #     MetadataCatalog.get("motorbike_bicycle_car_bus_val").thing_classes = ["motorbike_bicycle", "car_bus"]

    #     motorbike_bicycle_car_bus_weights = train_model(["motorbike_bicycle", "car_bus"], "output/motorbike_bicycle_car_bus", "motorbike_bicycle_car_bus")

    if os.path.exists("output/chair_sofa/model_final.pth"):
        chair_sofa_weights = "output/chair_sofa/model_final.pth"
    else:

        # Node for "chair" vs "sofa"

        register_coco_instances("chair_sofa_train", {}, "data/result/chair_sofa/annotations.json", "data/result/chair_sofa/images")
        register_coco_instances("chair_sofa_val", {}, "data/result/chair_sofa/annotations.json", "data/result/chair_sofa/images")

        MetadataCatalog.get("chair_sofa_train").thing_classes = ["chair", "sofa"]
        MetadataCatalog.get("chair_sofa_val").thing_classes = ["chair", "sofa"]

        chair_sofa_weights = train_model(["chair", "sofa"], "output/chair_sofa", "chair_sofa")
    
    if os.path.exists("output/cat_dog/model_final.pth"):
        cat_dog_weights = "output/cat_dog/model_final.pth"
    else:
            
        # Train leaf models
        # Node for "cat" vs "dog"

        register_coco_instances("cat_dog_train", {}, "data/result/cat_dog/annotations.json", "data/result/cat_dog/images")
        register_coco_instances("cat_dog_val", {}, "data/result/cat_dog/annotations.json", "data/result/cat_dog/images")

        MetadataCatalog.get("cat_dog_train").thing_classes = ["cat", "dog"]
        MetadataCatalog.get("cat_dog_val").thing_classes = ["cat", "dog"]

        cat_dog_weights = train_model(["cat", "dog"], "output/cat_dog", "cat_dog")

    if os.path.exists("output/cow_sheep/model_final.pth"):
        cow_sheep_weights = "output/cow_sheep/model_final.pth"
    else:

    # Node for "cow" vs "sheep"

        register_coco_instances("cow_sheep_train", {}, "data/result/cow_sheep/annotations.json", "data/result/cow_sheep/images")
        register_coco_instances("cow_sheep_val", {}, "data/result/cow_sheep/annotations.json", "data/result/cow_sheep/images")

        MetadataCatalog.get("cow_sheep_train").thing_classes = ["cow", "sheep"]
        MetadataCatalog.get("cow_sheep_val").thing_classes = ["cow", "sheep"]

        cow_sheep_weights = train_model(["cow", "sheep"], "output/cow_sheep", "cow_sheep")

    if os.path.exists("output/bus_car/model_final.pth"):
        bus_car_weights = "output/bus_car/model_final.pth"
    else:

        # Node for "bus" vs "car"

        register_coco_instances("bus_car_train", {}, "data/result/bus_car/annotations.json", "data/result/bus_car/images")
        register_coco_instances("bus_car_val", {}, "data/result/bus_car/annotations.json", "data/result/bus_car/images")

        MetadataCatalog.get("bus_car_train").thing_classes = ["bus", "car"]
        MetadataCatalog.get("bus_car_val").thing_classes = ["bus", "car"]

        bus_car_weights = train_model(["bus", "car"], "output/bus_car", "bus_car")

    if os.path.exists("output/bicycle_motorbike/model_final.pth"):
        bicycle_motorbike_weights = "output/bicycle_motorbike/model_final.pth"
    else:

        # Node for "bicycle" vs "motorbike"

        register_coco_instances("bicycle_motorbike_train", {}, "data/result/motorbike_bicycle/annotations.json", "data/result/motorbike_bicycle/images")
        register_coco_instances("bicycle_motorbike_val", {}, "data/result/motorbike_bicycle/annotations.json", "data/result/motorbike_bicycle/images")

        MetadataCatalog.get("bicycle_motorbike_train").thing_classes = ["motorbike", "bicycle"]
        MetadataCatalog.get("bicycle_motorbike_val").thing_classes = ["motorbike", "bicycle"]

        bicycle_motorbike_weights = train_model(["bicycle", "motorbike"], "output/bicycle_motorbike", "bicycle_motorbike")

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
        max_class_index = len(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes) - 1
        valid_classes = [i for i in classes if i <= max_class_index]
        #print("valid_classes", valid_classes)
        if not valid_classes:
            print("Warning: Detected class indices exceed the number of known classes.")
            return ["Unknown"]
        return [MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[i] for i in valid_classes]

    image = cv2.imread(image_path)

    #write to logs what each node is predicting


    root_prediction = predict_node(image, cfgs["root"], predictors["root"])
    


    # if len(root_prediction) > 0 and root_prediction[0] == "cat_dog_cow_sheep":
    #     #writelog
    #     with open("output/log.txt", "a") as log:
    #         log.write(f"Root prediction: {root_prediction[0]} and then ")
    #     cat_dog_cow_sheep_prediction = predict_node(image, cfgs["cat_dog_cow_sheep"], predictors["cat_dog_cow_sheep"])
    #     if len(cat_dog_cow_sheep_prediction) > 0 and cat_dog_cow_sheep_prediction[0] == "cat_dog":
    #         #writelog
    #         with open("output/log.txt", "a") as log:
    #             log.write(f"{cat_dog_cow_sheep_prediction[0]} and then ")
    #         cat_dog_prediction = predict_node(image, cfgs["cat_dog"], predictors["cat_dog"])
    #         return cat_dog_prediction[0] if len(cat_dog_prediction) > 0 else "Unknown"
    #     elif len(cat_dog_cow_sheep_prediction) > 0 and cat_dog_cow_sheep_prediction[0] == "cow_sheep":
    #         #writelog
    #         with open("output/log.txt", "a") as log:
    #             log.write(f"{cat_dog_cow_sheep_prediction[0]} and then ")
    #         cow_sheep_prediction = predict_node(image, cfgs["cow_sheep"], predictors["cow_sheep"])
    #         return cow_sheep_prediction[0] if len(cow_sheep_prediction) > 0 else "Unknown"
    # elif len(root_prediction) > 0 and root_prediction[0] == "motorbike_bicycle_car_bus":
    #     #writelog
    #     with open("output/log.txt", "a") as log:
    #         log.write(f"Root prediction: {root_prediction[0]} and then ")
    #     motorbike_bicycle_car_bus_prediction = predict_node(image, cfgs["motorbike_bicycle_car_bus"], predictors["motorbike_bicycle_car_bus"])
    #     if len(motorbike_bicycle_car_bus_prediction) > 0 and motorbike_bicycle_car_bus_prediction[0] == "motorbike_bicycle":
    #         #writelog
    #         with open("output/log.txt", "a") as log:
    #             log.write(f"{motorbike_bicycle_car_bus_prediction[0]} and then ")
    #         bicycle_motorbike_prediction = predict_node(image, cfgs["bicycle_motorbike"], predictors["bicycle_motorbike"])
    #         return bicycle_motorbike_prediction[0] if len(bicycle_motorbike_prediction) > 0 else "Unknown"
    #     elif len(motorbike_bicycle_car_bus_prediction) > 0 and motorbike_bicycle_car_bus_prediction[0] == "car_bus":
    #         #writelog
    #         with open("output/log.txt", "a") as log:
    #             log.write(f"{motorbike_bicycle_car_bus_prediction[0]} and then ")
    #         bus_car_prediction = predict_node(image, cfgs["bus_car"], predictors["bus_car"])
    #         return bus_car_prediction[0] if len(bus_car_prediction) > 0 else "Unknown"
    # elif len(root_prediction) > 0 and root_prediction[0] == "sofa_chair":
    #     #writelog
    #     with open("output/log.txt", "a") as log:
    #         log.write(f"Root prediction: {root_prediction[0]} and then ")
    #     chair_sofa_prediction = predict_node(image, cfgs["chair_sofa"], predictors["chair_sofa"])
    #     return chair_sofa_prediction[0] if len(chair_sofa_prediction) > 0 else "Unknown"
    # #writelog
    # with open("output/log.txt", "a") as log:
    #     log.write(f"Root prediction: Unknown")
    # return "Unknown"

    if len(root_prediction) > 0 and root_prediction[0] == "cat_dog":
        #writelog
        with open("output/log.txt", "a") as log:
            log.write(f"Root prediction: {root_prediction[0]} and then ")
        cat_dog_prediction = predict_node(image, cfgs["cat_dog"], predictors["cat_dog"])
        return cat_dog_prediction[0] if len(cat_dog_prediction) > 0 else "Unknown"
    elif len(root_prediction) > 0 and root_prediction[0] == "cow_sheep":
        #writelog
        with open("output/log.txt", "a") as log:
            log.write(f"Root prediction: {root_prediction[0]} and then ")
        cow_sheep_prediction = predict_node(image, cfgs["cow_sheep"], predictors["cow_sheep"])
        return cow_sheep_prediction[0] if len(cow_sheep_prediction) > 0 else "Unknown"
    elif len(root_prediction) > 0 and root_prediction[0] == "car_bus":
        #writelog
        with open("output/log.txt", "a") as log:
            log.write(f"Root prediction: {root_prediction[0]} and then ")
        bus_car_prediction = predict_node(image, cfgs["bus_car"], predictors["bus_car"])
        return bus_car_prediction[0] if len(bus_car_prediction) > 0 else "Unknown"
    elif len(root_prediction) > 0 and root_prediction[0] == "motorbike_bicycle":
        #writelog
        with open("output/log.txt", "a") as log:
            log.write(f"Root prediction: {root_prediction[0]} and then ")
        bicycle_motorbike_prediction = predict_node(image, cfgs["bicycle_motorbike"], predictors["bicycle_motorbike"])
        return bicycle_motorbike_prediction[0] if len(bicycle_motorbike_prediction) > 0 else "Unknown"
    elif len(root_prediction) > 0 and root_prediction[0] == "sofa_chair":
        #writelog
        with open("output/log.txt", "a") as log:
            log.write(f"Root prediction: {root_prediction[0]} and then ")
        chair_sofa_prediction = predict_node(image, cfgs["chair_sofa"], predictors["chair_sofa"])
        return chair_sofa_prediction[0] if len(chair_sofa_prediction) > 0 else "Unknown"
    #writelog
    with open("output/log.txt", "a") as log:
        log.write(f"Root prediction: Unknown")
    return "Unknown"





def load_weights():
    return {
        "normal": "output/normal/model_final.pth",	
        "root": "output/root/model_final.pth",
        #"cat_dog_cow_sheep": "output/cat_dog_cow_sheep/model_final.pth",
        #"motorbike_bicycle_car_bus": "output/motorbike_bicycle_car_bus/model_final.pth",
        "chair_sofa": "output/chair_sofa/model_final.pth",
        "cat_dog": "output/cat_dog/model_final.pth",
        "cow_sheep": "output/cow_sheep/model_final.pth",
        "bus_car": "output/bus_car/model_final.pth",
        "bicycle_motorbike": "output/bicycle_motorbike/model_final.pth"
        

    }

def register_and_set_classes():
    #if already registered, don't register again
    if "normal_train" in DatasetCatalog.list():
        return
    else:
        register_coco_instances("normal_train", {}, "data/result/normal/annotations.json", "data/result/normal/images")
        register_coco_instances("normal_val", {}, "data/result/normal/annotations.json", "data/result/normal/images")
        MetadataCatalog.get("normal_train").thing_classes = ["cat", "dog", "cow", "sheep", "motorbike", "bicycle", "car", "bus", "sofa", "chair"]
        MetadataCatalog.get("normal_val").thing_classes = ["cat", "dog", "cow", "sheep", "motorbike", "bicycle", "car", "bus", "sofa", "chair"]

    if "root_train" in DatasetCatalog.list():
        return
    else:
        register_coco_instances("root_train", {}, "data/result/root/annotations.json", "data/result/root/images")
        register_coco_instances("root_val", {}, "data/result/root/annotations.json", "data/result/root/images")
        MetadataCatalog.get("root_train").thing_classes = ["cat_dog", "cow_sheep", "motorbike_bicycle", "car_bus", "sofa_chair"]
        MetadataCatalog.get("root_val").thing_classes = ["cat_dog", "cow_sheep", "motorbike_bicycle", "car_bus", "sofa_chair"]
    
    # if "cat_dog_cow_sheep_train" in DatasetCatalog.list():
    #     return
    # else:
    #     register_coco_instances("cat_dog_cow_sheep_train", {}, "data/result/cat_dog_cow_sheep/annotations.json", "data/result/cat_dog_cow_sheep/images")
    #     register_coco_instances("cat_dog_cow_sheep_val", {}, "data/result/cat_dog_cow_sheep/annotations.json", "data/result/cat_dog_cow_sheep/images")
        
    #     MetadataCatalog.get("cat_dog_cow_sheep_train").thing_classes = ["cat_dog", "cow_sheep"]
    #     MetadataCatalog.get("cat_dog_cow_sheep_val").thing_classes = ["cat_dog", "cow_sheep"]
    
    # if "motorbike_bicycle_car_bus_train" in DatasetCatalog.list():
    #     return
    # else:
    #     register_coco_instances("motorbike_bicycle_car_bus_train", {}, "data/result/motorbike_bicycle_car_bus/annotations.json", "data/result/motorbike_bicycle_car_bus/images")
    #     register_coco_instances("motorbike_bicycle_car_bus_val", {}, "data/result/motorbike_bicycle_car_bus/annotations.json", "data/result/motorbike_bicycle_car_bus/images")
    #     MetadataCatalog.get("motorbike_bicycle_car_bus_train").thing_classes = ["motorbike_bicycle", "car_bus"]
    #     MetadataCatalog.get("motorbike_bicycle_car_bus_val").thing_classes = ["motorbike_bicycle", "car_bus"]

    if "chair_sofa_train" in DatasetCatalog.list():
        return
    else:
        register_coco_instances("chair_sofa_train", {}, "data/result/chair_sofa/annotations.json", "data/result/chair_sofa/images")
        register_coco_instances("chair_sofa_val", {}, "data/result/chair_sofa/annotations.json", "data/result/chair_sofa/images")
        MetadataCatalog.get("chair_sofa_train").thing_classes = ["chair", "sofa"]
        MetadataCatalog.get("chair_sofa_val").thing_classes = ["chair", "sofa"]

    if "cat_dog_train" in DatasetCatalog.list():
        return
    else:
        register_coco_instances("cat_dog_train", {}, "data/result/cat_dog/annotations.json", "data/result/cat_dog/images")
        register_coco_instances("cat_dog_val", {}, "data/result/cat_dog/annotations.json", "data/result/cat_dog/images")
        MetadataCatalog.get("cat_dog_train").thing_classes = ["cat", "dog"]
        MetadataCatalog.get("cat_dog_val").thing_classes = ["cat", "dog"]
    
    if "cow_sheep_train" in DatasetCatalog.list():
        return
    else:
        register_coco_instances("cow_sheep_train", {}, "data/result/cow_sheep/annotations.json", "data/result/cow_sheep/images")
        register_coco_instances("cow_sheep_val", {}, "data/result/cow_sheep/annotations.json", "data/result/cow_sheep/images")
        MetadataCatalog.get("cow_sheep_train").thing_classes = ["cow", "sheep"]
        MetadataCatalog.get("cow_sheep_val").thing_classes = ["cow", "sheep"]

    if "bus_car_train" in DatasetCatalog.list():
        return
    else:
        register_coco_instances("bus_car_train", {}, "data/result/bus_car/annotations.json", "data/result/bus_car/images")
        register_coco_instances("bus_car_val", {}, "data/result/bus_car/annotations.json", "data/result/bus_car/images")
        MetadataCatalog.get("bus_car_train").thing_classes = ["bus", "car"]
        MetadataCatalog.get("bus_car_val").thing_classes = ["bus", "car"]

    if "bicycle_motorbike_train" in DatasetCatalog.list():
        return
    else:
        register_coco_instances("bicycle_motorbike_train", {}, "data/result/motorbike_bicycle/annotations.json", "data/result/motorbike_bicycle/images")
        register_coco_instances("bicycle_motorbike_val", {}, "data/result/motorbike_bicycle/annotations.json", "data/result/motorbike_bicycle/images")
        MetadataCatalog.get("bicycle_motorbike_train").thing_classes = ["motorbike", "bicycle"]
        MetadataCatalog.get("bicycle_motorbike_val").thing_classes = ["motorbike", "bicycle"]



def test_accuracy():
    register_and_set_classes()
    print(MetadataCatalog.list())
    weights = load_weights()


    cfgs = {}
    predictors = {}
    
    for key, weight in weights.items():
        print("key: ", key)
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml"))
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
    validation_annotations_file = 'data/result/single_label/annotations.json'
    validation_images_folder = 'data/result/single_label/images'
    
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
