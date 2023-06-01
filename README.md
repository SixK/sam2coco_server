# sam2coco_server

This tool is a quick and dirty proof of concept base on facebook SAM intended to be used with coco-annotator (https://github.com/jsbroks/coco-annotator) to automate mask creation when labeling image.  

## Usage
Install python depencies:
>    pip install -r ./requirements.txt

Download SAM model:
>    download_model.sh


Start serv.py:
>    python3 serv.py

Start coco-annotator and connect to coco-annotator.
Create a new category named sam (or modify this line in serv.py to get annotation mask on another category -->  "name": "sam",)
Add "sam" category to the dataset you want.
Click "Image Settings" button
Modify "Annotate API" field: http://localhost:8000/
Click Close button
Click on "Annotate Image" button

All masks and annotations will be created under "sam" Category
As SAM can't predict mask class, you will have to handle each annotation manually to change Category or delete Annotations.

## Todo
- Filter Masks by score
- Modify DEXTR button (or add a SAM button using the same code) and modify serv.py to handle the same endpoints than for DEXTR.(In this mode, you would only have to point object you want to annotate.

