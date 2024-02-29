import attr
from typing import Any, Union, List, Tuple
from abc import ABCMeta, abstractmethod

import cv2
import torch
import numpy as np

from habitat import Config

import supervision as sv
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor


VisualObservation = Union[torch.Tensor, np.ndarray]


@attr.s(auto_attribs=True)
class Segment(metaclass=ABCMeta):
    config: Config
    
    def __attrs_post_init__(self):
        self._create_model(self.config)
    
    @abstractmethod
    def _create_model(self, config: Config) -> None:
        pass
    
    @abstractmethod
    def segment(self, image: VisualObservation, **kwargs) -> Any:
        pass
    

class GroundedSAM(Segment):
    def _create_model(self, config: Config) -> Any:
        GROUNDING_DINO_CONFIG_PATH = config.MAP.GROUNDING_DINO_CONFIG_PATH
        GROUNDING_DINO_CHECKPOINT_PATH = config.MAP.GROUNDING_DINO_CHECKPOINT_PATH
        SAM_CHECKPOINT_PATH = config.MAP.SAM_CHECKPOINT_PATH
        SAM_ENCODER_VERSION = config.MAP.SAM_ENCODER_VERSION
        device = torch.device("cuda", config.TORCH_GPU_ID if torch.cuda.is_available() else "cpu")
        
        self.grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=device
            )
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=device)
        self.sam_predictor = SamPredictor(sam)
        
    def _segment(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    
    def segment(self, image: VisualObservation, **kwargs) -> Tuple[np.ndarray, List[str], np.ndarray]:
        classes = kwargs.get("classes", [])
        box_threshold = kwargs.get("box_threshold", 0.35)
        text_threshold = kwargs.get("text_threshold", 0.25)
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = []
        
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        detections.mask = self._segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        
        for _, _, confidence, class_id, _ in detections:
            if class_id is not None:
                labels.append(f"{classes[class_id]} {confidence:0.2f}")
            else:
                labels.append(f"{classes[-1]} {confidence:0.2f}")

        # annotated_image.shape=(h,w,3)
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        
        return (detections.mask, labels, annotated_image)
    

class BatchWrapper:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a list of input images.
    """
    def __init__(self, model) -> None:
        self.model = model
    
    def __call__(self, images: List[VisualObservation]) -> List:
        return [self.model(image) for image in images]