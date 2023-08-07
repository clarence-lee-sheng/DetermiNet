
# DetermiNet
## Dataset Description
- **Paper:** [insert paper link here]()
- **Visit our github link here:** 
  - Clarence: clarence_leesheng@mymail.sutd.edu.sg 
  - Ganesh: m_ganeshkumar@u.nus.edu
- **Supervisor**
  - Cheston Tan: cheston-tan@i2r.a-star.edu.sg

### Dataset Summary

Determiners are an important word class that is used in the referencing and quantification of nouns. However existing datasets place less emphasis on determiners, compared to other word classes. Hence, we have designed the DetermiNet dataset, which is a visuolinguistic dataset comprising of the word class determiners. It comprises of 25 determiners with 10,000 examples each, totalling 250,000 samples. All scenes were synthetically generated using unity. The task is to predict bounding boxes to identify objects of interest, constrained by the semantics of the determiners  

<!-- , while state-of-the-art visual grounding models can achieve high detection accuracy, they are not designed to distinguish between all objects versus only certain objects of interest. Furthermore, existing datasets place much less emphasis on determiners, compared to other word classes. In order to address this, we have designed the DetermiNet dataset. -->

<div align="center">
  <figure>
    <figcaption>Figure 1. Samples of DetermiNet image-caption pairs, with their bounding box annotations and segmentations
    </figcaption>
    <br>
    <img src="./docs/assets/cover.png" width=530px/>
  </figure>
</div>

### Downloading the files: 
Download the files here [https://drive.google.com/drive/folders/1J5dleNxWvFUip5RBsTl6OqQBtpWO0r1k?usp=sharing](https://drive.google.com/drive/folders/1J5dleNxWvFUip5RBsTl6OqQBtpWO0r1k?usp=sharing) which contains: 

- The dataset in the images directory 
- COCO annotations and oracle tfrecords directory 
- oracle weights 
- real DetermiNet dataset

Setting up the files: 
- cd into root directory 
- extract the images folder and put it into data/ folder 
- extract annotations folder in root directory 
- put oracle_weights in oracle/ folder 

```
+ root 
--+ annotations 
-----+ oracle_tfrecords 
--+ data 
-----+ images 
--+ oracle
-----+ oracle_weights 
.
.
.
```

### Setup 
- Setup Unity Hub as per https://docs.unity3d.com/hub/manual/InstallHub.html
- Install Unity Editor 2021.3.9f1
- Install required packages
1. install requirements 
```
pip install -r requirements.txt 
pip install -e .
``` 
2. run setup.py
```
// assume in root directory of the repository 
cd DetermiNetUnity/Assets/utils
python setup_unity_and_clear_files.py
```

## Generating the dataset
Generate your own dataset by referring to the instructions [here](DATASET_SETUP.md)

### Dataset Structure

All data fields are based on the COCO annotation format. Refer to this link for more information: https://cocodataset.org/#home, We add the "input_oracle_annotations" field  to provide annotations for all bounding boxes per image to train the neurosymbolic model for DetermiNet. 

For all evaluation, metrics was measured in AP @ 0.5:0.95 based on [pycocotools](https://pypi.org/project/pycocotools/)

```
{
    'input_oracle_annotations':[
        {
            "id": ...,
            "image_id": ...,
            "category_id": ...,
            "area": ...,
            "bbox": ...,
            "iscrowd": ...
        }, 
        ...
    ],
    'annotations':[
        {
            "id": ...,
            "image_id": ...,
            "category_id": ...,
            "area": ...,
            "bbox": ...,
            "iscrowd": ...
        }, 
        ...
    ]
}
```

## Retraining the models 
### Training the Oracle model
```
python ./oracle/oracle_training.py --image_dir="data/" --tfrecords_dir="annotations/oracle_tfrecords/"
```

### MDETR/OFA
For [MDETR](https://github.com/ashkamath/mdetr) and [OFA](https://github.com/OFA-Sys/OFA), you may refer to the repositories in the link to run the baseline models. 

## Evaluation scripts 
### Evaluating baselines 
To reproduce the results we showed in the paper, you can run coco evaluation for mAP based on the jsons we generated in both ground_truths and predictions directories. 

You may run the following code to see the evaluation results for our Neurosymbolic model, MDETR and OFA 
```
cd evaluation 
python evaluation_baseline.py 
```

### Evaluating on new models using corrected ground truth 
To run on corrected evaluation first save your predictions as specified by the coco format in ./predictions as {model_name}_pred_results.json.

Afterwards, run the evaluation script as below, changing the model_name parameter to your desired name, this script will generate the corrected ground truth file under ground_truths using the model name and evaluate against the predictions in the predictions folder.

```
cd evaluation 
python evaluate.py --model_name=ns
```

### Real Dataset 
Download the real dataset [here](https://drive.google.com/drive/folders/1J5dleNxWvFUip5RBsTl6OqQBtpWO0r1k?usp=sharing)
<!-- 
### Citation Information

Provide the [BibTex](http://www.bibtex.org/)-formatted reference for the dataset. For example:
```
@article{article_id,
  author    = {Author List},
  title     = {Dataset Paper Title},
  journal   = {Publication Venue},
  year      = {2525}
}
```

If the dataset has a [DOI](https://www.doi.org/), please provide it here. -->