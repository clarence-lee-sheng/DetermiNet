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
```
pip install -r requirements.txt 
``` 

### Generating the dataset
#### Option 1: Through the Unity Editor 
1. open up the unity project under "DetermiNetProject" directory 
2. Setup the resolution to the image resolution that you want to generate (256 x 256)
![set screen resolution](./assets/screenResolution.png)
3. run setup.py
```
// assume in root directory of the repository 
cd DetermiNetProject/Assets/utils
python setup.py
```
4. Click play on the unity editor to run the generation script 