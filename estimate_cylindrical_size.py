from PIL import Image
from transformers import SAM3Processor, SAM3Model
import torch
import numpy as np

SAMMODEL = SAM3Model.from_pretrained('facebook/sam3').to('cuda')
SAMPROCESSOR = SAM3Processor.from_pretrained('facebook/sam3').to('cuda')


def segment_wheel_thread(image_path):
    image = Image.opn(image_path).convert('RGB')

    inputs = SAMPROCESSOR(images=image, text="wheel thread", return_tensors="pt").to('cuda')

    with torch.no_grad():
        outputs = SAMMODEL(**inputs)

    results = SAM3Processor.post_process_semantic_segmentation(
        outputs, 
        threshold = 0.9, 
        target_size = inputs.get('original_sizes'.tolist())
    )[0]

    mask_np = results.cpu().numpy()

    unique_ids = np.unique(mask_np)

    segment_ids = unique_ids[unique_ids > 0]

    if len(segment_ids) == 0:
        return None
    
    masks = [mask_np == seg_id for seg_id in segment_ids]

    return masks

