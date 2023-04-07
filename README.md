
# DetermiNet
## Dataset Description

- **Homepage:** [Add homepage URL here if available (unless it's a GitHub repository)]()
- **Repository:** https://github.com/Reckonzz/DetermiNet
- **Paper:** [If the dataset was introduced by a paper or there was a paper written describing the dataset, add URL here (landing page for Arxiv paper preferred)]()
- **Leaderboard:** -
- **Point of Contact:** Clarence, clarence_leesheng@mymail.sutd.edu.sg

### Dataset Summary

DetermiNet is a visuolinguistic dataset comprising of the word class determiners. It contains 25 determiners with 10000 examples each, totalling 250000 samples. All scenes were synthetically generated using unity.

### Supported Tasks and Leaderboards

For each of the tasks tagged for this dataset, give a brief description of the tag, metrics, and suggested models (with a link to their HuggingFace implementation if available). Give a similar description of tasks that were not covered by the structured tag set (repace the `task-category-tag` with an appropriate `other:other-task-name`).

- `task-category-tag`: The dataset can be used to train a model for [TASK NAME], which consists in [TASK DESCRIPTION]. Success on this task is typically measured by achieving a *high/low* [metric name](https://huggingface.co/metrics/metric_name). The ([model name](https://huggingface.co/model_name) or [model class](https://huggingface.co/transformers/model_doc/model_class.html)) model currently achieves the following score. *[IF A LEADERBOARD IS AVAILABLE]:* This task has an active leaderboard which can be found at [leaderboard url]() and ranks models based on [metric name](https://huggingface.co/metrics/metric_name) while also reporting [other metric name](https://huggingface.co/metrics/other_metric_name).

### Languages

All data is in the English language

## Dataset Structure

### Data Instances

As the ground truths for DetermiNet may contain multiple solutions. 

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

Provide any additional information that is not covered in the other sections about the data here. In particular describe any relationships between data points and if these relationships are made explicit.

### Data Fields

All data fields are based on the COCO annotation format. Refer to this link for more information: https://cocodataset.org/#home

### Data Splits

Train, test and validation splits are provided for the dataset. Stratified splitting was done to ensure an even distribution of determiner classes across all splits. All data and annotations were synthetically generated 

Provide the sizes of each split. As appropriate, provide any descriptive statistics for the features, such as average length.  For example:

|                         | train | validation | test |
|-------------------------|------:|-----------:|-----:|
| Input Sentences         |       |            |      |
| Average Sentence Length |       |            |      |

## Dataset Creation

### Curation Rationale

Determiners is an important word class used in referencing and quantification. Current visuolinguistic datasets do not sufficiently cover the word class determiners.

### Source Data and Annotations 

All data and annotations were synthetically generated using unity 


## Considerations for Using the Data

### Social Impact of Dataset

This dataset contains visuolinguistic data for determiners, which is lacking in many present day computer vision datasets. It serves as a diagnostic dataset to test the capabilities of grounded language models 

### Discussion of Biases

Provide descriptions of specific biases that are likely to be reflected in the data, and state whether any steps were taken to reduce their impact.

For Wikipedia text, see for example [Dinan et al 2020 on biases in Wikipedia (esp. Table 1)](https://arxiv.org/abs/2005.00614), or [Blodgett et al 2020](https://www.aclweb.org/anthology/2020.acl-main.485/) for a more general discussion of the topic.

If analyses have been run quantifying these biases, please add brief summaries and links to the studies here.


## Additional Information

### Dataset Curators

Clarence Lee Sheng: clarence_leesheng@mymail.sutd.edu.sg
Ganesh Kumar: ...
<!-- 
### Licensing Information

Provide the license and link to the license webpage if available. -->

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

If the dataset has a [DOI](https://www.doi.org/), please provide it here.

### Contributions

Thanks to [@github-username](https://github.com/<github-username>) for adding this dataset.



# DetermiNet

DetermiNet is a visuolinguistic dataset based on the word class "determiners". It aims to develop understanding of using determiners for referring expressions.

This is the code used to the DetermiNet dataset as described in the paper: 
https://openreview.net/pdf?id=UO1e-ReZ4IT
Presented at ____

You may use the code to generate synthetic images and text pairings to determiner phrases, such as the following: 

### DetermiNet Dataset 
![cover](./assets/cover.png)
### Setup 
- Setup Unity Hub as per https://docs.unity3d.com/hub/manual/InstallHub.html
- Install Unity Editor 2021.3.9f1
- Install required packages
1. install requirements 
```
pip install -r requirements.txt 
``` 
2. run setup.py
```
// assume in root directory of the repository 
cd DetermiNetProject/Assets/utils
python setup.py
```

## Generating the dataset
#### Through the Unity Editor 
1. open up the unity project under "DetermiNetProject" directory 
2. Setup the resolution to the image resolution that you want to generate (256 x 256) under the "Game" Tab 
![set screen resolution](./assets/screenResolution.png)
3. Set field of view to 75 on main camera and add ImageSynthesis component to it 
4. Create a 3D plane object as the floor with the following specifications: 
- position X: 0, Y: 0, Z: 0
- rotation X: 0, Y: 0, Z: 0
- scale X: 3.5, Y:3.5, Z: 3.5
5. Create a plane object as the scene background with the following specifications: 
- position X: 0, Y: 0, Z: 17
- rotation X: 270, Y: 0, Z: 0
- scale X: 12, Y:12, Z: 12
6. Give the planes a dark colour by applying the materials in the Resources/materials tab under Assets
7. Click play on the unity editor to run the generation script 


## Retraining the models 
### Neurosymbolic model
This section elaborates how you can rerun the Neuro-symbolic model which were explained 

### MDETR/OFA
For [MDETR](https://github.com/ashkamath/mdetr) and [OFA](https://github.com/OFA-Sys/OFA), you may refer to the repositories in the link to run the baseline models. we have also included the json files that we used in the preprocessed_data directory 

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

