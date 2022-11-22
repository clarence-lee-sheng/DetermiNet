from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval
import os 
import time

gt_dir = "../ground_truths"
prediction_dir = "../predictions"

models = ["ns", "mdetr", "ofa"]

for model in models: 
    print(model)
    cocoGt = COCO(os.path.join(gt_dir, f"mod_gt_{model}_annotations.json"))
    pred = cocoGt.loadRes(os.path.join(prediction_dir, f"{model}_pred_results.json"))
    cocoEval = COCOeval(cocoGt, pred, 'bbox')


    before = time.time() 
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    after = time.time()
    print(f"Time taken for {model}: {after-before}")