# check pytorch installation:
import torch, torchvision
import numpy as np

print(torch.__version__, torch.cuda.is_available())
#assert torch.__version__.startswith("1.9")  # please manually install torch 1.9 if Colab changes its default version

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy
import os, cv2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import log_first_n
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog,DatasetCatalog

from detectron2.data.datasets import register_coco_instances
# register_coco_instances("pendik_train", {}, "Config/label/mapilio_instances.json", "pendik/train")
# register_coco_instances("pendik_val", {}, "pendik/val/build.json", "pendik/val")
# register_coco_instances("coco_val", {}, "pothole/val/val.json", "coco/val")



cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"))
# cfg.DATASETS.TRAIN = ("pendik_train",)
# cfg.DATASETS.TEST = ()
# cfg.DATALOADER.NUM_WORKERS = 2
# # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("https://drive.google.com/file/d/1PEW7HTdO3bXQSiMUJYAjzZy8fSBk3Vda/view")  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 0.0005  # pick a good LR
# cfg.SOLVER.MAX_ITER = 30000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
# cfg.SOLVER.STEPS = []  # do not decay learning rate
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 445 # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()
# exit(1)
#
# # Inference should use the config with parameters that are used in training
# # cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join("//10.10.5.98/merge_dataset/inference", "multi_model_mask_rcnn_R_50_C4_3x.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
print(cfg.DATASETS.PROPOSAL_FILES_TRAIN)
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
import glob # noqa
from tqdm import tqdm

# for index, img_path in enumerate(tqdm(glob.glob(os.path.join('mapilio_inference', 'images', '*.jpg')))):
for index, img_path in enumerate(tqdm(glob.glob(os.path.join('images', '*')))):
# for index, img_path in enumerate(tqdm(glob.glob(os.path.join('mapilio_inference', 'visio_images', '*.jpeg')))):
    basename = os.path.basename(img_path)
    im = cv2.imread(img_path)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata= {},
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )

    classes=outputs['instances'].to("cpu").pred_classes
    print(classes)
#    reshaped_masks = np.reshape(masks,((masks.shape[1]),(masks.shape[2])))
#    print(reshaped_masks.shape)
#    cv2.imshow("Circular Mask", reshaped_masks)
#    x1,y1,x2,y2 = boxes[0,:]
#    im=cv2.rectangle(im,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
#    im=cv2.resize(im,(2400,1200))
#    cv2.imshow('Show',im)
#    print(x1,y1,x2,y2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    image_path = f'{index}_{basename}'
#    cv2.imwrite(os.path.join('result', image_path), output.get_image()[:, :, ::-1])
    cv2.imshow('win',out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    # exit(1)