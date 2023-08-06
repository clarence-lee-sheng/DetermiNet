import argparse
import json
import sys
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append("..")
from utils.eval_utils import *

parser = argparse.ArgumentParser(description='Running evaluation on corrected ground truth')

parser.add_argument("--model_name", help="sets the model to evaluation", default="ns")

args = parser.parse_args()

model_name = args.model_name

results_dir = "predictions"
split = "bb_test"
# json.dump(results, open(os.path.join("ns_results", f"{split}_results.json"), "w"))

annType = "bbox"
dataDir = "../"

if args.model_name == "ns":
    annFile = f"{dataDir}/annotations/annotations_test.json"
else: 
    annFile = f"{dataDir}/annotations/annotations_v1_test.json"
resFile = f"{dataDir}/annotations/predictions/{model_name}_pred_results.json"

cocoGt = COCO(annFile)
cocoDt = cocoGt.loadRes(resFile)
annType = "bbox"

cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.imgIds = sorted(cocoGt.getImgIds())
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

generate_corrected_gt_json(gt_dir=annFile, results_dir=resFile, model_name=model_name)

modannFile = f"{dataDir}/ground_truths/mod_gt_{model_name}_annotations.json"
modcocoGt = COCO(modannFile)
cocoDt = modcocoGt.loadRes(resFile)

print('After correcting annotations')
cocoEval = COCOeval(modcocoGt, cocoDt, annType)
cocoEval.params.imgIds = sorted(modcocoGt.getImgIds())
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()