
# DetermiNet
## Dataset Description

- **Homepage:** [Add homepage URL here if available (unless it's a GitHub repository)]()
- **Repository:** https://github.com/Reckonzz/DetermiNet
- **Paper:** [insert paper linke here]()
- **Point of Contact:** Clarence, clarence_leesheng@mymail.sutd.edu.sg and Ganesh, m_ganeshkumar@u.nus.edu

### Dataset Summary

DetermiNet is a visuolinguistic dataset comprising of the word class determiners. It contains 25 determiners with 10000 examples each, totalling 250000 samples. All scenes were synthetically generated using unity.

### Supported Tasks and Leaderboards

DetermiNet is a grounded language task focussed on generating bounding boxes based on referencing and quantification from determiners

Metrics is measured in AP @ 0.5:0.95 based on [pycocotools](https://pypi.org/project/pycocotools/)

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

### Data Fields

All data fields are based on the COCO annotation format. Refer to this link for more information: https://cocodataset.org/#home, We add the "input_oracle_annotations" field to train the neurosymbolic model for DetermiNet

### Data Splits

Train, test and validation splits are provided for the dataset. Stratified splitting was done to ensure an even distribution of determiner classes across all splits. All data and annotations were synthetically generated 

Provide the sizes of each split. As appropriate, provide any descriptive statistics for the features, such as average length.  For example:

|                         | train | validation | test |
|-------------------------|------:|-----------:|-----:|
| Input Sentences         |       |            |      |
| Average Sentence Length |       |            |      |

## Dataset Creation

### Curation Rationale

Determiners is an important word class used in referencing and quantification. Current visuolinguistic datasets do not sufficiently cover the word class determiners. We seek to cover this gap by creating a comprehensive large scale dataset which covers a large range of determiners. 

### Source Data and Annotations 

All data and annotations were synthetically generated using unity 

## Considerations for Using the Data

### Social Impact of Dataset

This dataset contains visuolinguistic data for determiners, which is lacking in many present day computer vision datasets. It serves as a diagnostic dataset to test the capabilities of grounded language models on the determiners word class. 

### Discussion of Biases

The generation of the dataset was based on our definitions (refer to Figure 1) in our paper. We provide configuration files to adjust parameters to your preferences 

## Additional Information

### Dataset Curators

Clarence Lee Sheng: clarence_leesheng@mymail.sutd.edu.sg
Ganesh Kumar: m_ganeshkumar@u.nus.edu

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

