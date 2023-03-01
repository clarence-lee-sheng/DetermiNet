using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading;
using System.Threading.Tasks;
using UnityEditor;
using System.IO;
using System;
using System.Linq;
using Newtonsoft.Json;

public class DeterminerGenerator : MonoBehaviour 
{
    public ObjectPool pool; 
    public Dictionary<string, Dictionary<string, List<int>>>countableQuantities;
    public Dictionary<string, Dictionary<string, List<float>>> uncountableQuantities;
    public COCOAnnotations cc_inputs;
    public COCOAnnotations cc_outputs;
    public Dictionary<string, Category> categories;
    public Dictionary<string, string> pluralForms;

    


    public DeterminerGenerator(ObjectPool pool, Dictionary<string, Dictionary<string, List<int>>> countableQuantities, Dictionary<string, Dictionary<string, List<float>>> uncountableQuantities, COCOAnnotations cc_inputs, COCOAnnotations cc_outputs, Dictionary<string, Category> categories)
    {
        this.pool = pool;
        this.countableQuantities = countableQuantities;
        this.uncountableQuantities = uncountableQuantities;
        this.cc_inputs = cc_inputs;
        this.cc_outputs = cc_outputs;
        this.categories = categories;
        
        string pluralJSONString = File.ReadAllText($"{Application.dataPath}/Scripts/config/plural.json");

        this.pluralForms = JsonConvert.DeserializeObject<Dictionary<string, string>>(pluralJSONString);
    }

    public async Task generate(int n_examples, string determiner, List<Scene> scenes)
    {
        SceneConfig sceneConfig; 
        Debug.Log($"Generating {n_examples} examples of {determiner}...");
        switch (determiner)
        {
            case "a":
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "countablesConsonants",
                    numSelectedRange: new List<int>(){1,1}, 
                    nExtra: 2,
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("none")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "an":
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "countablesVowels",
                    numSelectedRange: new List<int>(){1,1}, 
                    nExtra: 2,
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("none")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "my":
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "all",
                    numSelectedRange: new List<int>(){1,3}, 
                    nExtra: 0,
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn(
                        spawns: new List<string> {"containerMain"}, 
                        characterConfig: new CharacterConfig(false)
                    ),
                    unselectedMainObjSpawns: new Spawn(
                        spawns: new List<string> {"containerSecondary", "table"}, 
                        characterConfig: new CharacterConfig(false)
                    )
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "your":
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "all",
                    numSelectedRange: new List<int>(){1,3}, 
                    nExtra: 0,
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn(
                        spawns: new List<string> {"containerSecondary"}, 
                        characterConfig: new CharacterConfig(false)
                    ),
                    unselectedMainObjSpawns: new Spawn(
                        spawns: new List<string> {"containerMain", "table"}, 
                        characterConfig: new CharacterConfig(false)
                    )
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "our": 
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "all",
                    numSelectedRange: new List<int>(){1,5}, 
                    nExtra: 2,
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn(
                        spawns: new List<string> {"containerMain", "containerSecondary"}, 
                        characterConfig: new CharacterConfig(false)
                    ),
                    unselectedMainObjSpawns: new Spawn(
                        spawns: new List<string> {"table"}, 
                        characterConfig: new CharacterConfig(false)
                    )
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "all": 
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "all",
                    numSelectedRange: new List<int>(){2,5}, 
                    nExtra: 0,
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("none")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "every": 
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "countables",
                    numSelectedRange: new List<int>(){3,5}, 
                    nExtra: 0,
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("none")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "any":
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "all",
                    numSelectedRange: new List<int>(){1,1}, 
                    nExtra: 4,
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("none")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "this": 
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "all",
                    numSelectedRange: new List<int>(){1,1}, 
                    nExtra: 2,
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn(
                        new List<string> {"containerMain", "table"},
                        new CharacterConfig("characterMain", 6f, true)
                    ), 
                    unselectedMainObjSpawns: new Spawn(
                        spawns: new List<string> {"containerSecondary", "table"},
                        characterConfig: new CharacterConfig("characterMain", 10f, false)
                    )
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "that": 
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "all",
                    numSelectedRange: new List<int>(){1,1}, 
                    nExtra: 2,
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn(
                        new List<string> {"containerSecondary", "table"},
                        new CharacterConfig("characterMain", 10f, false)
                    ), 
                    unselectedMainObjSpawns: new Spawn(
                        spawns: new List<string> {"containerMain", "table"},
                        characterConfig: new CharacterConfig("characterMain", 6f, true)
                    )
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "these": 
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "countables",
                    numSelectedRange: new List<int>(){2,4}, 
                    nExtra: 0,
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn(
                        new List<string> {"containerMain", "table"},
                        new CharacterConfig("characterMain", 6f, true)
                    ), 
                    unselectedMainObjSpawns: new Spawn(
                        spawns: new List<string> {"containerSecondary", "table"},
                        characterConfig: new CharacterConfig("characterMain", 10f, false)
                    )
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "those": 
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "countables",
                    numSelectedRange: new List<int>(){2,4}, 
                    nExtra: 0,
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn(
                        new List<string> {"containerSecondary", "table"},
                        new CharacterConfig("characterMain", 10f, false)
                    ), 
                    unselectedMainObjSpawns: new Spawn(
                        spawns: new List<string> {"containerMain", "table"},
                        characterConfig: new CharacterConfig("characterMain", 6f, true)
                    )
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "some": 
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "all",
                    numSelectedRange: new List<int>(){1,1}, 
                    nExtra: 2, 
                    useQuantityConfig: true,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("none")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "many": 
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "countables",
                    numSelectedRange: new List<int>(){8,9}, 
                    nExtra: 0, 
                    useQuantityConfig: true,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("none")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "few": 
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "countables",
                    numSelectedRange: new List<int>(){2,3}, 
                    nExtra: 0, 
                    useQuantityConfig: true,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("none")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "both": 
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "countables",
                    numSelectedRange: new List<int>(){2,2}, 
                    nExtra: 0, 
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("none")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "either":
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "countables",
                    numSelectedRange: new List<int>(){1,1}, 
                    nExtra: 0, 
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("all")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "neither":
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "countables",
                    numSelectedRange: new List<int>(){2,2}, 
                    nExtra: 0, 
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("none")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "little": 
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "uncountables",
                    numSelectedRange: new List<int>(){1,4}, 
                    nExtra: 0, 
                    useQuantityConfig: true,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("all")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "much": 
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "uncountables",
                    numSelectedRange: new List<int>(){1,4}, 
                    nExtra: 0, 
                    useQuantityConfig: true,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("all")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "no": 
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "all",
                    numSelectedRange: new List<int>(){1,5}, 
                    nExtra: 0,
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("none")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "the": 
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "all",
                    numSelectedRange: new List<int>(){1,1}, 
                    nExtra: 0,
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("none")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "half":
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "countables",
                    numSelectedRange: new List<int>(){1,3}, 
                    nExtra: 0,
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("all")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "several": 
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "countables",
                    numSelectedRange: new List<int>(){1,1}, 
                    nExtra: 2, 
                    useQuantityConfig: true,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("none")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
            case "each": 
                sceneConfig = new SceneConfig(
                    totalObjCounts: new List<int>{10,12,14,16,18},
                    mainObjSet: "countables",
                    numSelectedRange: new List<int>(){2,5}, 
                    nExtra: 0,
                    useQuantityConfig: false,
                    selectedMainObjSpawns: new Spawn("all"),
                    unselectedMainObjSpawns: new Spawn("none")
                );
                await spawnAndCapture(n_examples, determiner, scenes, sceneConfig);
                break;
        }
    }

    public async Task spawnAndCapture(int n_examples, string determiner, List<Scene> scenes, SceneConfig sceneConfig)
    {
        Character selectedCharacter;
        Character unselectedCharacter; 
        int numMainObjsUnselected; 
        int counter = 0; 
        List<EnvObject> selectedMainObjSpawns;
        List<EnvObject> unselectedMainObjSpawns;
        List<EnvObject> allSpawns;
        int numMainObjsSpawned; 
        int numEl;
        List<int> numSelectedRange;
        bool mainObjIsCountable, mainObjIsLiquid;
        List<float> liquidRange;

        List<string> mainObjSet = new List<string>();
        List<string> mainObjSetCopy = new List<string>(pool.objectSets[sceneConfig.mainObjSet]);

        mainObjSet = pool.objectSets[sceneConfig.mainObjSet];

        double nExamplesPerScene = Math.Ceiling((float) n_examples / (36f * mainObjSet.Count));

        for (int j = 0; j < nExamplesPerScene; j++)
        {
            foreach(Scene scene in scenes)
            {
                scene.SetActive(); 
                selectedMainObjSpawns = new List<EnvObject>();
                unselectedMainObjSpawns = new List<EnvObject>();
                allSpawns = new List<EnvObject>(){scene.objects["containerMain"], scene.objects["containerSecondary"], scene.objects["table"]};
                foreach(string key in sceneConfig.selectedMainObjSpawns.spawns)
                {
                    selectedMainObjSpawns.Add(scene.objects[key]);
                };
                foreach(string key in sceneConfig.unselectedMainObjSpawns.spawns)
                {
                    unselectedMainObjSpawns.Add(scene.objects[key]);
                };


                foreach(int quantity in sceneConfig.totalObjCounts)
                {
                    foreach(string mainObjName in mainObjSet)
                    {
                        //spawn mainObjects 
                        if (counter >= n_examples)
                        {
                            break;
                        }
                        mainObjIsCountable = categories[mainObjName].supercategory.Split("_")[0] == "countable"; 
                        if (mainObjIsCountable)
                        {
                            mainObjIsLiquid = false;
                            if (sceneConfig.useQuantityConfig)
                            {
                                numSelectedRange = countableQuantities[mainObjName][determiner];
                            }else{
                                numSelectedRange = sceneConfig.numSelectedRange;
                            }
                            
                        }else{
                            mainObjIsLiquid = categories[mainObjName].supercategory.Split("_")[1] == "liquid";
                            numSelectedRange = sceneConfig.numSelectedRange;
                        }

                        if (mainObjIsLiquid && sceneConfig.useQuantityConfig)
                        {
                            liquidRange = uncountableQuantities[mainObjName][determiner];
                        }else{
                            liquidRange = new List<float>(){0.1f,1f};
                        }
                        numMainObjsSpawned = UnityEngine.Random.Range(numSelectedRange[0], numSelectedRange[1] + sceneConfig.nExtra + 1);
                                
                        selectedCharacter = new Character(
                            scene.objects[sceneConfig.selectedMainObjSpawns.characterConfig.name], 
                            sceneConfig.selectedMainObjSpawns.characterConfig
                        );
                        pool.spawnInScene(numMainObjsSpawned, new List<string>(){mainObjName}, selectedMainObjSpawns, selectedCharacter, "target", liquidRange, true);
                       

                        // add some unlabelled objects 
                        if (sceneConfig.unselectedMainObjSpawns.spawns.Count == 0 && determiner != "little" && determiner !="much")
                        {
                            numMainObjsUnselected = 0; 
                        }
                        else
                        {
                            unselectedCharacter = new Character(
                                scene.objects[sceneConfig.unselectedMainObjSpawns.characterConfig.name], 
                                sceneConfig.unselectedMainObjSpawns.characterConfig
                            );
                            if(determiner == "half") {
                                numMainObjsUnselected = numMainObjsSpawned;
                            }else if(determiner == "either" ) {
                                numMainObjsUnselected = 1;
                            }else{
                                numMainObjsUnselected = UnityEngine.Random.Range(1, 4);
                            }
                            
                            if (mainObjIsLiquid && (determiner == "little" || determiner == "much"))
                            {
                                pool.spawnInScene(numMainObjsUnselected, new List<string>(){mainObjName}, unselectedMainObjSpawns, unselectedCharacter, "notTarget", liquidRange, false);
                            } else {
                                pool.spawnInScene(numMainObjsUnselected, new List<string>(){mainObjName}, unselectedMainObjSpawns, unselectedCharacter, "notTarget", liquidRange, true);
                            }
                            
                        }

                        //spawn non main objects 
                            Character character = new Character(
                                scene.objects["characterMain"], 
                                new CharacterConfig(false)
                            );
                            if (quantity - numMainObjsSpawned - numMainObjsUnselected > 0)
                            {
                                List<string> othersObjectSet = new List<string> (pool.objectSets["all"]);
                                if (!(mainObjIsLiquid && sceneConfig.useQuantityConfig))
                                {
                                    othersObjectSet.Remove(mainObjName);
                                }
                                
                                
                                pool.spawnInScene(quantity - numMainObjsSpawned - numMainObjsUnselected, othersObjectSet, allSpawns, character, "notTarget", liquidRange, !(mainObjIsLiquid && sceneConfig.useQuantityConfig));  
                            }


                        List<int> randomIndexes = new List<int>();

                        if (new List<string>{"all", "every", "our", "my", "your","no", "each", "these", "those", "half"}.Contains(determiner) || (!mainObjIsCountable && new List<string>{"some", "little", "much"}.Contains(determiner)))
                        {
                            numEl = numMainObjsSpawned;
                        }else{
                            numEl = UnityEngine.Random.Range(numSelectedRange[0], Math.Min(numSelectedRange[1], numMainObjsSpawned) + 1);
                        }

                        while (randomIndexes.Count < numEl)
                        {
                            int randIdx = UnityEngine.Random.Range(0, pool.active["target"].Count); 
                            while (randomIndexes.Contains(randIdx))
                            {
                                randIdx = UnityEngine.Random.Range(0, pool.active["target"].Count); 
                            }
                            randomIndexes.Add(randIdx);
                        }
                        foreach (int idx in randomIndexes)
                        {
                            GameObject selected = pool.active["target"][idx].gameObject;
                            selected.layer = LayerMask.NameToLayer("selected");
                            foreach (Transform child in selected.transform)
                            {
                                child.gameObject.layer = LayerMask.NameToLayer("selected");
                            }
                        }
                        // }

                        // await Task.Delay(10);
                        int id, image_id, category_id, area, iscrowd;
                        string imgFilepath = $"{Application.streamingAssetsPath}/{Constants.datasetFolder}/images/{determiner}";
                        string segmentationFilepath = $"{Application.streamingAssetsPath}/{Constants.datasetFolder}/segmentations/{determiner}";
                        image_id = cc_inputs.images.Count;
                        string objNameWithPlularity;
                        if ((numEl > 1 && determiner != "each") || determiner == "half") {
                            objNameWithPlularity = pluralForms[mainObjName];
                        }else {
                            objNameWithPlularity = mainObjName;
                        }
                        string filename = $"{image_id}_{determiner}_{objNameWithPlularity}";
                        
                        List<int> bbox; 

                        

                        string caption = $"{determiner} {objNameWithPlularity}"; 
                        List<EnvObject> allObjects = new List<EnvObject>(pool.active["target"]);
                        allObjects.AddRange(pool.active["notTarget"]);

                        //cc_inputs is the oracle bounding boxes: e.g. it contains annotations of all bounding boxes 
                        //cc_outputs is the determiner labelled bounding boxes: e.g. it contains annotations of a subset of the bounding boxes based on the determiners

                        // add the coco annotations for all game objects 
                        foreach (EnvObject envObj in allObjects)
                        {
                            GameObject obj = envObj.gameObject;
                            image_id = image_id;
                            category_id = categories[obj.name.Replace("(Clone)", "")].id; ;
                            bbox = Helper.getBoundingBox(obj.GetComponent<Rigidbody>(), Camera.main);
                            area = bbox[2] * bbox[3];
                            iscrowd = 0;

                            cc_inputs.input_oracle_annotations.Add(new OracleSegmentationAnnotation(
                                cc_inputs.input_oracle_annotations.Count,
                                image_id,
                                category_id,
                                area,
                                iscrowd, 
                                bbox, 
                                envObj.liquidLevel
                            ));

                            //save in output if object is labelled by the determiner phrase 

                            if (obj.layer != 0)
                            {
                                cc_inputs.annotations.Add(new SegmentationAnnotation(
                                    cc_inputs.annotations.Count,
                                    image_id,
                                    category_id,
                                    area,
                                    iscrowd, 
                                    bbox 
                                ));

                                // cc_outputs.annotations.Add(new SegmentationAnnotation(
                                //     cc_outputs.annotations.Count,
                                //     image_id,
                                //     category_id,
                                //     area,
                                //     iscrowd, 
                                //     bbox 
                                // ));
                            }
                        }
                      
                        EnvObject mainTray = scene.objects["containerMain"];
                        EnvObject secondaryTray = scene.objects["containerSecondary"];
                        category_id = categories["tray"].id;
                        List<int> mainTrayBbox = Helper.getBoundingBox(mainTray.gameObject.GetComponent<Rigidbody>(), Camera.main);
                        List<int> secondaryTrayBbox = Helper.getBoundingBox(secondaryTray.gameObject.GetComponent<Rigidbody>(), Camera.main);
                        int mainTrayArea = mainTrayBbox[2] * mainTrayBbox[3];
                        int secondaryTrayArea = secondaryTrayBbox[2] * secondaryTrayBbox[3];
                        iscrowd = 0;

                        //add the coco annotations for the two trays 
                        cc_inputs.input_oracle_annotations.Add(new OracleSegmentationAnnotation(
                            cc_inputs.input_oracle_annotations.Count,
                            image_id,
                            category_id,
                            mainTrayArea,
                            iscrowd, 
                            mainTrayBbox, 
                            -1 
                        ));

                        cc_inputs.input_oracle_annotations.Add(new OracleSegmentationAnnotation(
                            cc_inputs.input_oracle_annotations.Count,
                            image_id,
                            category_id,
                            secondaryTrayArea,
                            iscrowd, 
                            secondaryTrayBbox,
                            -1
                        ));


                        cc_inputs.images.Add(new Image(image_id, $"images/{determiner}/{filename}_img.png", caption));
                        cc_inputs.segmentation_images.Add(new Image(image_id, $"segmentations/{determiner}/{filename}_img.png", caption));
                        // cc_outputs.images.Add(new Image(image_id, $"segmentations/{determiner}/{filename}_img.png"));
                        // cc_inputs.phrase_annotations.Add(new PhraseAnnotation(cc_inputs.phrase_annotations.Count, image_id, caption));
                        // cc_outputs.phrase_annotations.Add(new PhraseAnnotation(cc_outputs.phrase_annotations.Count, image_id, caption));

                        counter += 1;
                        
                        Helper.synthTakeSnapshot(0, imgFilepath, filename);
                        Helper.synthTakeSnapshot(2, segmentationFilepath, filename);
                        await Task.Delay(1);
                        pool.reset();
                        await Task.Delay(1);

                        // if (numMainObjsUnselected != numEl) {
                        //     Debug.Log("error");
                        //     Debug.Log($"numMainObjsUnselected: {numMainObjsUnselected}, numEl: {numEl}");
                        // }
                        

                    }  

                }
                scene.SetInactive();     
            }
        }
    }

    
}