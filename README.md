# YOLO

Using yolo with botsort to track objects. 


## process

1. The object were labeled using groundingDino: 

https://medium.com/@hadarpinhas/streamlining-dataset-labeling-with-groundingdino-a-local-inference-approach-be0e3f83c423

Also, split train/val/test in split_data.py 

2. after the labeling yolo was trained and weights saved, see train_yolo.py

3. Detecting and tracking and counting objects, see per_img_inference.py. inference.py inputs and outputs a video.

The tracker type is defined in 

"...\AppData\Local\Programs\Python\Python310\Lib\site-packages\ultralytics\solutions\object_counter.py"

        class ObjectCounter(BaseSolution):
            def __init__(self, **kwargs):
            """Initializes the ObjectCounter class for real-time object counting in video streams."""
            super().__init__(**kwargs)

->

"...\AppData\Local\Programs\Python\Python310\Lib\site-packages\ultralytics\solutions\solutions.py"

        self.CFG = {**DEFAULT_SOL_DICT, **DEFAULT_CFG_DICT}

->

"...\AppData\Local\Programs\Python\Python310\Lib\site-packages\ultralytics\utils\__init__.py" 

        DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)

->

"...\AppData\Local\Programs\Python\Python310\Lib\site-packages\ultralytics\cfg\default.yaml":

# Tracker settings ------------------------------------------------------------------------------------------------------
tracker: botsort.yaml # (str) tracker type, choices=[botsort.yaml, bytetrack.yaml]