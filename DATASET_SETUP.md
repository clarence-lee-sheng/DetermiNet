### Setup
The files generated will go into the StreamingAssets/dataset folder. Running the below script will clear all files inside the StreamingAssets folder and setup theh directories for generation 

```
cd DetermiNetUnity/Assets/utils
python setup_unity_and_clear_files.py
```
### Option 1: Use precreated scene 
1. open up the unity project under "DetermiNetProject" directory 
2. Setup the resolution to the image resolution that you want to generate (256 x 256) under the "Game" Tab 
![set screen resolution](./assets/screenResolution.png)
3. Go under Assets/Scenes and select DetermiNetScene.unity file 
4. click play to run the generation 

### Option 2: Setup the scene yourself
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

### Edit parameters for determiner generation
Configuration files can be found at DetermiNetUnity/Assets/Scripts/config 
- categories.json: List of all the object categories and indicate if it is an uncountable or countable quantity 
- countableQuantities.json: Set the number of objects for countables for each determiner in the range [low, high]
- uncountableQuantities.json: Set the number of objects for uncountables for each determiner in the range [low, high]
- plural.json: List of plural forms for each noun 

```
# Example elements in countableQuantities.json 
{
    "apple": {
        "few": [2,3], #few apples mean 2-3 apples spawned 
        "some": [4,5], #some apples mean 4-5 apples spawned 
        "many": [8,9], #many apples mean 8-9 apples spawned 
        "several": [6,7] #several apples mean 6-7 apples spawned 
    },
    "onion": {
        "few": [2,3], #few onions mean 2-3 onions spawned
        "some": [4,5], #some onions mean 4-5 onions spawned 
        "many": [8,9], #many onions mean 8-9 onions spawned 
        "several": [6,7] #several onions mean 6-7 onions spawned 
    },
...
}

Example elements in uncountableQuantities.json 
{
    "grape juice": {
        "little": [0.1,0.2],  #little grape juice mean 10%-20% of the glass filled  
        "some": [0.5,0.6], #some grape juice mean 50%-60% of the glass filled  
        "much": [0.9,1.0] #much grape juice mean 90%-100% of the glass filled  
    },
    "cranberry juice": {
        "little": [0.1,0.2], #little cranberry juice mean 10%-20% of the glass filled
        "some": [0.5,0.6], #some cranbery juice mean 50%-60% of the glass filled  
        "much": [0.9,1.0] #much cranberry juice mean 90%-100% of the glass filled 
    },
...
}

```