from autodistill_efficientsam import EfficientSAM
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = EfficientSAM(ontology=CaptionOntology({"book": "book"}))

# predict on an image
result = base_model.predict("bookshelf.jpeg", confidence=0.1)

import supervision as sv

image = cv2.imread("bookshelf.jpeg")

mask_annotator = sv.MaskAnnotator()
annotated_frame = mask_annotator.annotate(
	scene=image.copy(),
	detections=result,
)

sv.plot_image(annotated_frame)