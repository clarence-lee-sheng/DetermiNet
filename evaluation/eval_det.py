#from backend.utils import saveload
import numpy as np
#from backend.utils import compute_all_ious, create_output_txt, center_coord
import itertools
import os
import operator
import json
from collections import defaultdict


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    if np.isnan(iou):
        print(iou)
    # return the intersection over union value
    return iou

def xywh_to_coord(boxes):
    return np.concatenate([boxes[..., :2], boxes[..., :2] + boxes[..., 2:]], axis=-1)

def coord_to_xywh(boxes):
    return np.concatenate([boxes[..., :2], boxes[..., 2:] - boxes[..., :2]], axis=-1)


def compute_all_ious(gtbb,predbb):
    allious = np.zeros([len(gtbb), len(predbb)])
    for i in range(len(gtbb)):
        for j in range(len(predbb)):
            allious[i,j] = bb_intersection_over_union(xywh_to_coord(gtbb[i]), xywh_to_coord(predbb[j]))
    return allious

def center_coord(boxes):
    return np.concatenate([boxes[..., :2] + boxes[..., 2:]/2], axis=-1)


# Should be everything except, all, every, much, little, your, our, my

# custom a evaluation
# - image: 3 apples, caption 'a apple', output: any 1 apple
# 1) get ground truth with all apples bounding box
# 2) compute IoU between predicted apple and all apples
# 3) select apple with highest IoU as ground truth and discard the rest. dont modify prediction

import itertools
import operator

determiners = ["a", "an", "all", "any", "every", "my", "your", "this", "that", "these", "those", "some", "many",
                   "few", "both", "neither", "little", "much", "either", "our", "no", "several", "half", "each",
                   "the"]

categories = []
cats = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../annotations/annotations_test.json')))['categories']
for cat in cats: 
    categories.append(cat['name'])

def generate_corrected_gt_json(gt_dir, results_dir,mod_res_dir, max_bboxes=20,sometype='new'):
    print('Ensure prediction bounding boxes are in xywh format!')
    results = json.load(open(os.path.join(results_dir)))
    inputs = json.load(open(os.path.join(gt_dir)))
    input_annotations = inputs["input_oracle_annotations"]
    image_idx = defaultdict(int)

    for i, image in enumerate(inputs["images"]):
        image_idx[image['id']] = i

    determiners = ["a", "an", "all", "any", "every", "my", "your", "this", "that", "these", "those", "some", "many",
                   "few", "both", "neither", "little", "much", "either", "our", "no", "several", "half", "each",
                   "the"]
    objects = []
    get_attr = operator.attrgetter('image_id')
    #     print(results)
    img_id_grouped = [list(g) for k, g in itertools.groupby(results, lambda x: x["image_id"])]
    res_bboxes = list(map(lambda arr: list(map(lambda y: np.array(y["bbox"]), arr)), img_id_grouped))
    res_scores = list(map(lambda arr: list(map(lambda y: y["score"], arr)), img_id_grouped))
    res_labels = list(map(lambda arr: list(map(lambda y: y["category_id"], arr)), img_id_grouped))

    bboxes = [[] for i in range(len(inputs["images"]))]
    scores = [[] for i in range(len(inputs["images"]))]
    labels = [[] for i in range(len(inputs["images"]))]

    for i, group in enumerate(img_id_grouped):
        image_id = group[0]["image_id"]
        j = image_idx[image_id]
        bboxes[j] += res_bboxes[i]
        scores[j] += res_scores[i]
        labels[j] += res_labels[i]

    for i, group in enumerate(bboxes):
        for j in range(len(group), max_bboxes):
            bboxes[i].append(np.array([0, 0, 0, 0]))
            scores[i].append(0)
            labels[i].append(0)
        bboxes[i] = np.array(bboxes[i])
        scores[i] = np.array(scores[i])
        labels[i] = np.array(labels[i])
    bboxes = np.array(bboxes)
    scores = np.array(scores)
    labels = np.array(scores)

    categories = inputs["categories"]
    gt_annotations = inputs["annotations"]
    gt_bboxes = [[] for i in inputs["images"]]
    input_bboxes = [[] for i in inputs["images"]]

    for ann in gt_annotations:
        gt_bboxes[image_idx[ann["image_id"]]].append(np.array(ann["bbox"]))
        #print(image_idx[ann["image_id"]])
    for ann in input_annotations:
        category_one_hot = [0 for i in range(len(categories))]
        category_one_hot[ann["category_id"]] = 1
        input_bboxes[image_idx[ann["image_id"]]].append(np.array(ann["bbox"] + [1] + category_one_hot))

    for i, group in enumerate(gt_bboxes):
        for j in range(len(group), max_bboxes):
            gt_bboxes[i].append(np.array([0, 0, 0, 0]))

    for i, group in enumerate(input_bboxes):
        for j in range(len(group), max_bboxes):
            input_bboxes[i].append(np.array([0, 0, 0, 0] + [0] + [0 for i in range(len(categories))]))

    input_bboxes = np.array([np.array(row) for row in input_bboxes])
    gt_bboxes = np.array([np.array(row) for row in gt_bboxes])

    categories = inputs["categories"]
    category_one_hot = [0 for i in range(len(categories))]
    captions = []
    count = 0
    for i, img in enumerate(inputs["images"]):
        caption = img["caption"]
        det = caption.split()[0]
        # if det not in ["a", "an"]:
        #     count += 1 
        #     # print(caption)
        #     print(img)
        #     if count > 10: 
        #         raise Exception("Det not a or an")
        noun = " ".join(caption.split()[1:])
        determiners.index(det)
        noun_id = 0
        if noun[-1] == "s":
            noun = noun[:-1]
        for cat in categories:
            if cat["name"] == noun:
                noun_id = cat["id"]
        det_one_hot = [0 for i in range(len(determiners))]
        det_one_hot[determiners.index(det)] = 1
        noun_one_hot = [0 for i in range(len(categories))]
        noun_one_hot[noun_id] = 1
        caption_one_hot = det_one_hot + noun_one_hot

        captions.append(caption_one_hot)

    captions = np.array(captions)
    print("ground truth bboxes shape: ", gt_bboxes.shape)
    print("input bboxes shape: ", input_bboxes.shape)
    print("pred bboxes shape: ", bboxes.shape)
    print("scores shape: ", scores.shape)
    print("captions shape", captions.shape)

    gt_bboxes = main_change_gt_multiple_soln(gt_bboxes, input_bboxes, captions, scores, bboxes, max_bboxes, sometype)
    new_bboxes = []
    new_annotations = []
    count = 0
    for i, img_bboxes in enumerate(gt_bboxes):
        #print(inputs["images"][i]["id"])
        for bbox in img_bboxes:
            bbox = list(bbox)
            bbox = list(map(np.int, bbox))
            if bbox == [0, 0, 0, 0]:
                continue
            else:
                new_bboxes.append(bbox)
                new_annotations.append({
                    "id": count,
                    "image_id": inputs["images"][i]["id"],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "category_id": 1,
                    "iscrowd": 0
                })
            count += 1
    new_json = {
        #"licenses": inputs["licenses"],
        "categories": inputs["categories"],
        "images": inputs["images"],
        "annotations": new_annotations
    }

    print('saving modified ground truth')
    json.dump(new_json, open(mod_res_dir, "w"))


def main_change_gt_multiple_soln(gt_bbox, input_bb, input_cap, pred_score, pred_bb, max_bb,sometype):
    assert gt_bbox.shape[0] == pred_bb.shape[0]
    gt_bb = np.copy(gt_bbox[:,:, :4])
    scorethres = 0.1
    #gt_bb = my_your_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=[5,6])
    gt_bb = a_an_either_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=[0, 1, 18,24])
    gt_bb = any_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=3, scorethres=scorethres)
    gt_bb = this_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=7)
    gt_bb = that_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=8)
    #gt_bb = these_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=9)
    #gt_bb = those_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=10)
    # new
    if sometype == 'new':
        gt_bb = some_many_few_several_half_changegt(
            max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=[11,12,13, 21,22],  lowerbound=[4,8,2,6,-1],
            upperbound=[5,9,3,7,-1], scorethres=scorethres)
    else:
        gt_bb = some_many_few_several_half_changegt(
            max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=[11,12,13, 21,22],  lowerbound=[5,8,2,4,-1],
            upperbound=[6,9,3,7,-1], scorethres=scorethres)
    print('done correcting ground truth')
    return gt_bb


def some_many_few_several_half_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb,detidx=[11,12,13,21],
                                        lowerbound=[5,8,2,4], upperbound=[6,9,3,7], scorethres=0.1):
    for det, lowerb, upperb in zip(detidx, lowerbound, upperbound):
        # Version 1: some 5-6, many 8-9, few 2-3, several 4-7; # Version 2: some 4-5, many 8-9, few 2-3, several 6-7
        alldet = np.argmax(input_cap[:,:25],axis=1) == det # captions with some
        allcountobj = np.argmax(input_cap[alldet, 25:], axis=1) < 10  # captions with countables
        # index of captions with some and countables
        nidx = np.arange(len(gt_bb))[alldet]
        nidx = nidx[allcountobj]
        for n in nidx:
            objroi = np.argmax(input_cap[n,25:])
            presentobj = input_bb[n,:,4]>0
            objidx = np.argmax(input_bb[n,presentobj,5:],axis=1) == objroi

            # all correct answers
            allobjbb = input_bb[n,presentobj,:4][objidx] #* 256

            if lowerb == -1 and upperb == -1:
                lb = up = len(allobjbb)//2
            else:
                lb, up = lowerb, upperb

            # logic: use predictions with score above 0.5 for any
            critanyidx = np.arange(max_bb)[pred_score[n]>scorethres]
            predbb = pred_bb[n]#, critanyidx]

            # check if predicted bb IoU with all 'apple' bb
            allious = compute_all_ious(allobjbb, predbb)
            if len(allious)<1:
                print(objidx)
            gtbbsel = np.argmax(allious,axis=0)
            gtbbsel = np.unique(gtbbsel)
            gtbb = allobjbb[gtbbsel]

            if not lb<=len(gtbb)<=up:
                if len(gtbb) < lb:
                    # print(len(gtbb), gtbb, lb)
                    remainsoln = np.delete(np.arange(len(allobjbb)), gtbbsel)
                    # print("n remaining: ", len(remainsoln), remainsoln)
                    idx = np.random.choice(remainsoln, lb-len(gtbb), replace=False)
                    gtbb = np.concatenate([gtbb, allobjbb[idx]],axis=0) # atleast 5 bounding boxes
                else:
                    #descpred = np.argsort(pred_score[n])[::-1][:up]
                    #gtbb = gtbb[descpred]
                    gtbb = gtbb[:up]

            assert (lb-1)<len(gtbb)<(up+1), print('some, many, few, several wrong gt')
            pad_gtbb = np.pad(gtbb, ((0, max_bb - len(gtbb)), (0, 0)))[None,:]
            gt_bb[n] = pad_gtbb
            # if np.array_equal(pred_bb, gt_bb) is False:
            #     np.sum(pred_bb != gt_bb)
    return gt_bb


def a_an_either_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=[0,1,18]):
    # a, an, either
    # print("Data: ")
    # for bb in input_bb[:4]: 
    #     print("inp_bb", bb[:4])
    # print("GT: ")
    # print("length",len(gt_bb))
    # for bb in gt_bb[:4]: 
    #     print("gt",bb[0])

    for aidx in detidx:
        alla = np.argmax(input_cap[:,:25],axis=1) == aidx
        nidx = np.arange(len(gt_bb))[alla]
        for n in nidx:
            # print(n)
            # print(input_cap[n])
            objroi = np.argmax(input_cap[n,25:])
            # print(determiners[np.argmax(input_cap[n,25:])])
            # print(categories[objroi])
            presentobj = input_bb[n,:,4]>0

            # print("objroi", objroi)
            # print("presentobj", presentobj)
            objidx = np.argmax(input_bb[n,presentobj,5:],axis=1) == objroi

            # all correct answers
            allobjbb = input_bb[n,presentobj,:4][objidx] #* 256
            # print(input_bb[n, presentobj])
            # print("objidx",objidx)
            # print(input_bb)
            # print("bb", allobjbb)

            lb = 1
            up = 1

            # Logic: use top 1 prediction for a
            #top1idx = np.argsort(pred_score[n])[::-1][:1]
            predbb = pred_bb[n]#, top1idx]
            # print("pred", pred_bb[n])

            # check if predicted bb IoU with all 'apple' bb
            allious = compute_all_ious(allobjbb, predbb)

            # print(allobjbb, predbb)
            if len(allious)<1:
                print(objidx)
            # gtbbsel = np.argmax(allious,axis=0)
            # gtbbsel = np.unique(gtbbsel)
            # gtbb = allobjbb[gtbbsel]

            gtbbsel = np.any(allious, axis=1)  # remove 0s
            # gtbbsel = np.unique(gtbbsel)
            gtbb = allobjbb[gtbbsel]
            # print("GT", gtbb)

            #assert len(gtbb) == 1
            if not lb <= len(gtbb) <= up:
                if len(gtbb) < lb:
                    remainsoln = np.delete(np.arange(len(allobjbb)), gtbbsel)
                    # print(np.arange(len(allobjbb)))
                    # print(remainsoln)
                    idx = np.random.choice(remainsoln, lb - len(gtbb), replace=False)
                    gtbb = np.concatenate([gtbb, allobjbb[idx]], axis=0)  # atleast 5 bounding boxes
                else:
                    # descpred = np.argsort(pred_score[n])[::-1][:up]
                    # gtbb = gtbb[descpred]
                    gtbb = gtbb[:up]
            pad_gtbb = np.pad(gtbb, ((0, max_bb - len(gtbb)), (0, 0)))[None,:]
            gt_bb[n] = pad_gtbb
            # if np.array_equal(pred_bb, gt_bb) is False:
            #     np.sum(pred_bb != gt_bb)
    return gt_bb

#create_output_txt(gdt=gt_bb, predt=pred_bb, confi=pred_score,gd_cls=input_cap[:,:20],pred_cls=pred_cls, directory='ns_cls_gta/mod_gt')


def both_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=14):
    # both
    alldet = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(gt_bb))[alldet]
    for n in nidx:
        objroi = np.argmax(input_cap[n, 25:])
        presentobj = input_bb[n, :, 4] > 0
        objidx = np.argmax(input_bb[n, presentobj, 5:], axis=1) == objroi

        # all correct answers
        allobjbb = input_bb[n, presentobj, :4][objidx]  # * 256

        # Logic: use top 2 prediction for a
        top2idx = np.argsort(pred_score[n])[::-1][:2]
        predbb = pred_bb[n, top2idx]

        # check if predicted bb IoU with all 'apple' bb
        allious = compute_all_ious(allobjbb, predbb)
        if len(allious) < 1:
            print(objidx)
        gtbbsel = np.argmax(allious,axis=0)
        gtbbsel = np.unique(gtbbsel)
        gtbb = allobjbb[gtbbsel]
        assert len(gtbb) == 2
        pad_gtbb = np.pad(gtbb, ((0, max_bb - len(gtbb)), (0, 0)))[None,:]
        gt_bb[n] = pad_gtbb
        # if np.array_equal(pred_bb, gt_bb) is False:
        #     np.sum(pred_bb != gt_bb)
    return gt_bb

def any_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=3, scorethres=0.1):
    # any
    allany = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(gt_bb))[allany]
    for n in nidx:
        objroi = np.argmax(input_cap[n, 25:])
        presentobj = input_bb[n, :, 4] > 0
        objidx = np.argmax(input_bb[n, presentobj, 5:], axis=1) == objroi

        # all correct answers
        allobjbb = input_bb[n, presentobj, :4][objidx]  # * 256

        lb = 1
        up = max(len(allobjbb)-1,1)

        # logic: use predictions with score above 0.5 for any
        critanyidx = np.arange(max_bb)[pred_score[n]>scorethres]
        predbb = pred_bb[n]#, critanyidx]

        # check if predicted bb IoU with all 'apple' bb
        allious = compute_all_ious(allobjbb, predbb)
        if len(allious) < 1:
            print(objidx)

        gtbbsel = np.any(allious,axis=1)  # remove 0s
        #gtbbsel = np.unique(gtbbsel)
        gtbb = allobjbb[gtbbsel]

        if not lb <= len(gtbb) <= up:
            if len(gtbb) < lb:
                remainsoln = np.delete(np.arange(len(allobjbb)), gtbbsel)
                idx = np.random.choice(remainsoln, lb - len(gtbb), replace=False)
                gtbb = np.concatenate([gtbb, allobjbb[idx]], axis=0)  # atleast 5 bounding boxes
            else:
                # descpred = np.argsort(pred_score[n])[::-1][:up]
                # gtbb = gtbb[descpred]
                gtbb = gtbb[:up]

        assert (lb - 1) < len(gtbb) < (up + 1), print('some, many, few, several wrong gt')
        pad_gtbb = np.pad(gtbb, ((0, max_bb - len(gtbb)), (0, 0)))[None, :]
        gt_bb[n] = pad_gtbb
    return gt_bb


def this_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=7):
    ## this
    allthis = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(gt_bb))[allthis]
    for n in nidx:
        objroi = np.argmax(input_cap[n, 25:])
        presentobj = input_bb[n, :, 4] > 0
        objidx = np.argmax(input_bb[n, presentobj, 5:], axis=1) == objroi

        # all correct answers
        allobjbb = input_bb[n, presentobj, :4][objidx]  # * 256

        # logic: object closer to camera with highest prediction score
        #center = np.array([128.0,256.0])
        this_thres = 113  # object must be bottom half of RGB image
        #objdist = np.linalg.norm(center - center_coord(allobjbb) , axis=1)

        correct_this_bbs = allobjbb [allobjbb[:,1] > this_thres]
        if len(correct_this_bbs)<1:
            correct_this_bbs = gt_bb[n,input_bb[n, :, 4] > 0,:4]

        #target_bb = output_bb[n,:,:4]

        top1idx = np.argsort(pred_score[n])[::-1][:1]
        predbb = pred_bb[n, top1idx]

        # check if predicted bb IoU with all 'apple' bb
        allious = compute_all_ious(correct_this_bbs, predbb)
        if len(allious) < 1:
            print(predbb)
            print(correct_this_bbs)
        gtbbsel = np.argmax(allious,axis=0)
        gtbbsel = np.unique(gtbbsel)
        gtbb = allobjbb[gtbbsel]
        assert len(gtbb) == 1
        pad_gtbb = np.pad(gtbb, ((0, max_bb - len(gtbb)), (0, 0)))[None,:]
        gt_bb[n] = pad_gtbb
        # if np.array_equal(pred_bb, gt_bb) is False:
        #     np.sum(pred_bb != gt_bb)
    return gt_bb


def that_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=9):
    ## that
    alldet = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(gt_bb))[alldet]
    for n in nidx:
        objroi = np.argmax(input_cap[n, 25:])
        presentobj = input_bb[n, :, 4] > 0
        objidx = np.argmax(input_bb[n, presentobj, 5:], axis=1) == objroi

        # all correct answers
        allobjbb = input_bb[n, presentobj, :4][objidx]  # * 256

        # logic: object closer to camera with highest prediction score
        #center = np.array([128.0,256.0])
        this_thres = 113  # object must be bottom half of RGB image

        #objdist = np.linalg.norm(center - center_coord(allobjbb), axis=1)

        correct_that_bbs = allobjbb [allobjbb[:,1] < this_thres]
        if len(correct_that_bbs)<1:
            correct_that_bbs = gt_bb[n,input_bb[n, :, 4] > 0,:4]

        #target_bb = output_bb[n,:,:4]

        top1idx = np.argsort(pred_score[n])[::-1][:1]
        predbb = pred_bb[n, top1idx]

        # check if predicted bb IoU with all 'apple' bb
        allious = compute_all_ious(correct_that_bbs, predbb)
        if len(allious) == 0:
            print(predbb)
            print(correct_that_bbs)
        gtbbsel = np.argmax(allious,axis=0)
        gtbbsel = np.unique(gtbbsel)
        gtbb = allobjbb[gtbbsel]
        assert len(gtbb) == 1
        pad_gtbb = np.pad(gtbb, ((0, max_bb - len(gtbb)), (0, 0)))[None,:]
        gt_bb[n] = pad_gtbb
        # if np.array_equal(pred_bb, gt_bb) is False:
        #     np.sum(pred_bb != gt_bb)
    return gt_bb




#create_output_txt(gdt=gt_bb, predt=pred_bb, confi=pred_score,gd_cls=input_cap[:,:20],pred_cls=pred_cls, directory='ns_cls_modgt/mod_gt_a_an_either_any_both_this_that_these_those_2')


if __name__ == '__main__':

    # gt_bb = generate_corrected_gt_json(gt_dir='../data_model/annotations/test_annotations.json', results_dir='../data_model/ns_results/test_results.json')


    determiners = ["a", "an", "all", "any", "every", "my", "your", "this", "that", "these", "those", "some", "many",
                   "few", "both", "neither", "little", "much", "either", "our", "no", "several", "half", "each",
                   "the"]
    #[pred_bb, pred_score, input_bb, input_cap, output_bb] = saveload('load','../data_model/test_predbb_only_v3', 1)
    #[pred_bb, pred_cls, pred_score, input_bb, input_cap, output_bb] = saveload('load','../data_model/test_predbb_out_v3', 1)
    #[pred_bb,pred_cls, pred_score, input_bb, input_cap, output_bb] = saveload('load','../data_model/test_predbb_det_v3',1)

    #create_output_txt(gdt=output_bb[:,:,:4], predt=pred_bb, confi=pred_score, directory='../data_model/ns_gt_bb/ori_gt')
    #create_output_txt(gdt=output_bb[:, :, :4], predt=pred_bb, confi=pred_score ,gd_cls=input_cap[:,:25],pred_cls=pred_cls,directory='../data_model/ns_cls_gt/ori_gt')

    #gt_bb = np.copy(output_bb[:,:,:4])  # modified ground truth bounding box labels

    #modgt_bb = main_change_gt_multiple_soln(gt_bb, input_bb, input_cap, pred_score, pred_bb)

    #create_output_txt(gdt=gt_bb, predt=pred_bb, confi=pred_score, directory='../data_model/ns_gt_bb/mod_gt')
    #create_output_txt(gdt=gt_bb, predt=pred_bb, confi=pred_score,gd_cls=input_cap[:,:25],pred_cls=pred_cls,directory='../data_model/ns_cls_gt/mod_gt')

'''
def these_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=9):
    ## these
    allthese = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(gt_bb))[allthese]
    for n in nidx:
        objroi = np.argmax(input_cap[n, 25:])
        presentobj = input_bb[n, :, 4] > 0
        objidx = np.argmax(input_bb[n, presentobj, 5:], axis=1) == objroi

        # all correct answers
        allobjbb = input_bb[n, presentobj, :4][objidx]  # * 256

        # logic: object closer to camera with highest prediction score
        #center = np.array([128.0,256.0])
        these_thres = 113  # object must be bottom half of RGB image

        #objdist = np.linalg.norm(center - center_coord(allobjbb), axis=1)

        correct_these_bbs = allobjbb [allobjbb[:,1] > these_thres]
        if len(correct_these_bbs)<1:
            correct_these_bbs = gt_bb[n,input_bb[n, :, 4] > 0,:4]
        #target_bb = output_bb[n,:,:4]

        critthatidx = np.arange(max_bb)[pred_score[n]>0.5]
        predbb = pred_bb[n, critthatidx]

        # check if predicted bb IoU with all 'apple' bb
        allious = compute_all_ious(correct_these_bbs, predbb)
        if len(allious) == 0:
            print(predbb)
            print(correct_these_bbs)
        gtbbsel = np.argmax(allious,axis=0)
        gtbbsel = np.unique(gtbbsel)
        gtbb = allobjbb[gtbbsel]

        # if len(correct_that_bbs) != len(predbb):
        #     print(np.argmax(input_bb[n, :, 5:], axis=1))
        #     print(len(correct_that_bbs), len(predbb))
        #     print('next')

        if len(gtbb)<2:
            remainsoln = np.delete(np.arange(len(correct_these_bbs)), gtbbsel)
            if len(remainsoln)<1:
                print(gtbbsel)
            idx = np.random.choice(remainsoln,1,replace=False)
            gtbb = np.concatenate([gtbb, correct_these_bbs[idx]],axis=0)
            #print('adding soln to these')

        assert len(gtbb) >1
        pad_gtbb = np.pad(gtbb, ((0, max_bb - len(gtbb)), (0, 0)))[None,:]
        gt_bb[n] = pad_gtbb
    return gt_bb


def those_changegt(max_bb, gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=10):
    ## those
    allthose = np.argmax(input_cap[:,:25],axis=1) == detidx
    nidx = np.arange(len(gt_bb))[allthose]
    for n in nidx:
        objroi = np.argmax(input_cap[n, 25:])
        presentobj = input_bb[n, :, 4] > 0
        objidx = np.argmax(input_bb[n, presentobj, 5:], axis=1) == objroi

        # all correct answers
        allobjbb = input_bb[n, presentobj, :4][objidx]  # * 256

        # logic: object closer to camera with highest prediction score
        # center = np.array([128.0,256.0])
        # this_thres = 128  # object must be bottom half of RGB image
        those_thres = 113

        #objdist = np.linalg.norm(center - center_coord(allobjbb), axis=1)

        correct_those_bbs = allobjbb [allobjbb[:,1] < those_thres]
        if len(correct_those_bbs)<1:
            correct_those_bbs = gt_bb[n,input_bb[n, :, 4] > 0,:4]

        #target_bb = output_bb[n,:,:4]

        critthatidx = np.arange(max_bb)[pred_score[n]>0.5]
        predbb = pred_bb[n, critthatidx]

        # check if predicted bb IoU with all 'apple' bb
        allious = compute_all_ious(correct_those_bbs, predbb)
        if len(allious) == 0:
            print(predbb)
            print(correct_those_bbs)
        gtbbsel = np.argmax(allious,axis=0)
        gtbbsel = np.unique(gtbbsel)
        gtbb = allobjbb[gtbbsel]

        if len(gtbb)<2:
            remainsoln = np.delete(np.arange(len(correct_those_bbs)), gtbbsel)
            idx = np.random.choice(remainsoln,1,replace=False)
            gtbb = np.concatenate([gtbb, correct_those_bbs[idx]],axis=0)
            #print('adding soln to those')
        assert len(gtbb) > 1
        pad_gtbb = np.pad(gtbb, ((0, max_bb - len(gtbb)), (0, 0)))[None,:]
        gt_bb[n] = pad_gtbb
    return gt_bb
'''
# your
# anyidx = 6
# allany = np.argmax(input_cap[:,:20],axis=1) == anyidx
# nidx = np.arange(len(pred_bb))[allany]
# for n in nidx:
#     objroi = np.argmax(input_cap[n,20:])
#     objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi
#
#     # all targets
#     allobjbb = input_bb[n,:,:4][objidx] * 256
#     # target of interest: base of item on tray opposite
#     ythes = 128
#     targets = allobjbb[allobjbb[:,1]<ythes]
#
#     #target_bb = output_bb[n,:,:4]
#
#     gtbb = targets
#     pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
#     gt_bb[n] = pad_gtbb


# all santiy check that code is correct
# anyidx = 2
# allany = np.argmax(input_cap[:,:20],axis=1) == anyidx
# nidx = np.arange(len(pred_bb))[allany]
# for n in nidx:
#     objroi = np.argmax(input_cap[n,20:])
#     objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi
#
#     # all coorect answers
#     allobjbb = input_bb[n,:,:4][objidx] * 256
#
#     # logic: use predictions with score above 0.5 for any
#     critanyidx = np.arange(20)[pred_score[n]>0.5]
#     predbb = pred_bb[n, critanyidx]
#
#     if len(allobjbb) != len(predbb):
#         print(np.argmax(input_bb[n, :, 5:], axis=1))
#         print(len(allobjbb), len(predbb))
#         print('next')
#
#     # check if predicted bb IoU with all 'apple' bb
#     allious = compute_all_ious(allobjbb, predbb)
#     gtbbsel = np.arange(len(allobjbb)) #np.argmax(allious,axis=1)
#     gtbb = allobjbb[gtbbsel]
#     pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
#     gt_bb[n] = pad_gtbb


'''
def my_your_changegt(gt_bb, input_bb, input_cap, pred_score, pred_bb, detidx=[5,6]):
    # a, an, either
    for aidx in detidx:
        alla = np.argmax(input_cap[:,:25],axis=1) == aidx
        nidx = np.arange(len(pred_bb))[alla]
        for n in nidx:
            objroi = np.argmax(input_cap[n,25:])
            objidx = np.argmax(input_bb[n,:,5:],axis=1) == objroi

            # all correct answers
            allobjbb = input_bb[n,:,:4][objidx] #* 256

            # tray coord
            trayidx = np.argmax(input_bb[n,:,5:],axis=1) == 15
            traycoord = input_bb[n,:,:4][trayidx]
            traydist = np.linalg.norm(center_coord(traycoord),axis=1)
            mytray = traycoord[np.argmax(traydist)]
            yourtray = traycoord[np.argmin(traydist)]

            # Logic: use top 1 prediction for a
            top1idx = np.argsort(pred_score[n])[::-1][:1]
            predbb = pred_bb[n, top1idx]

            # check if predicted bb IoU with all 'apple' bb
            allious = compute_all_ious(allobjbb, predbb)
            if len(allious)<1:
                print(objidx)
            gtbbsel = np.argmax(allious,axis=0)
            gtbbsel = np.unique(gtbbsel)
            gtbb = allobjbb[gtbbsel]
            #assert len(gtbb) == 1
            if len(gtbb) !=1:
                print(gtbb)
            pad_gtbb = np.pad(gtbb, ((0, 20 - len(gtbb)), (0, 0)))[None,:]
            gt_bb[n] = pad_gtbb
    return gt_bb
'''