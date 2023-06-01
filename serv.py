from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import base64
from pycocotools import coco
import cv2
from PIL import Image
import numpy as np
import json

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

sam = sam_model_registry["vit_b"](checkpoint="model/sam_vit_b_01ec64.pth")
device="cuda"
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

def masksToAnnotation(masks):
    annotations=[]
    id=0
    for mask in masks:
        contours, _ = cv2.findContours(mask["segmentation"].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Convert the contour to the format required for segmentation in COCO format
        segmentation = []
        for contour in contours:
            contour = contour.flatten().tolist()
            contour_pairs = [(contour[i], contour[i+1]) for i in range(0, len(contour), 2)]
            segmentation.append([int(coord) for pair in contour_pairs for coord in pair])

        # Define annotation in COCO format
        annotation = {
            "id": id,
            "image_id": 0,
            "category_id": 1,
            "segmentation": segmentation,
            "width": 800,
            "height": 600,
            "area": int(cv2.contourArea(contours[0])),
            "bbox": [int(x) for x in cv2.boundingRect(contours[0])],
            "metadata": {},
            "color": "#f4311f",
            "iscrowd": 0
        }
        annotations.append(annotation)
        id+=1

    return annotations

def createCoco(annotations):
    coco='''
{
    "coco": {
        "categories": [
            {
                "id": 1,
                "name": "sam",
                "supercategory": null,
                "metadata": {},
                "color": "#1c6bfb"
            }
        ],
        "images": [
            {
                "id": 0,
                "width": 800,
                "height": 600,
                "file_name": "",
                "path": "",
                "license": null,
                "fickr_url": null,
                "coco_url": null,
                "date_captured": null,
                "metadata": {}
            }
        ],
        "annotations": '''+json.dumps(annotations)+'''
        }
    }'''
    return coco

app = Flask(__name__)
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
})

@app.route('/', methods=['POST'])
def process_image():
    print(dir(request.files))
    # img_file = request.files['file']
    img_file = request.files['image']

    print(img_file)
    
    img = Image.open(img_file.stream).convert('RGB')
    print('img:', img.width, img.height)
    img = np.asarray(img)
    print(img)
            
    masks = mask_generator.generate(img)
    annot=masksToAnnotation(masks)

    print(annot)
    coco=createCoco(annot)

    print("coco:", coco)
    return json.loads(coco)


if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)
