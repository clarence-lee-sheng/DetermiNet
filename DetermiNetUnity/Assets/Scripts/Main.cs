using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading;
using System.Threading.Tasks;
using UnityEditor;
using System.IO;
using System;
using Newtonsoft.Json;
public class Main : MonoBehaviour
{
    public Camera[] allCameras;
    // Start is called before the first frame update
    async void Start()
    {
        // RectTransform rt = (RectTransform)tablePrefab.transform;
        COCOAnnotations cc_inputs = new COCOAnnotations();
        COCOAnnotations cc_outputs = new COCOAnnotations();

        string categoriesJSONString = File.ReadAllText($"{Application.dataPath}/Scripts/config/categories.json");
        string countableQuantitiesJSONString = File.ReadAllText($"{Application.dataPath}/Scripts/config/countableQuantities.json");
        string uncountableQuantitiesJSONString = File.ReadAllText($"{Application.dataPath}/Scripts/config/uncountableQuantities.json");

        // categories contains the coco categories information for all objects 
        Dictionary<string, Category> categories = JsonConvert.DeserializeObject<Dictionary<string, Category>>(categoriesJSONString);

        // countable quantities contain the number of objects for few, some, many
        Dictionary<string, Dictionary<string, List<int>>> countableQuantities = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, List<int>>>>(countableQuantitiesJSONString);

        // uncountable quantities contain the number of objects for little, some, many
        Dictionary<string, Dictionary<string, List<float>>> uncountableQuantities = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, List<float>>>>(uncountableQuantitiesJSONString);

        foreach(KeyValuePair<string, Category> cat in categories)
        {
            cc_inputs.categories.Add(cat.Value);
            cc_outputs.categories.Add(cat.Value);
        }
        
        // save the different groups of objects into an Assets object 
        Rigidbody[] countablesVowelsRigidBodies = Resources.LoadAll<Rigidbody>($"{Constants.countablesVowelsPath}");
        Rigidbody[] countablesConsonantsRigidBodies = Resources.LoadAll<Rigidbody>($"{Constants.countablesConsonantsPath}");
        Rigidbody[] uncountablesRigidBodies = Resources.LoadAll<Rigidbody>($"{Constants.uncountablesPath}");

        // create the object pool for the different groups of objects
        ObjectPool objectPool = new ObjectPool(countablesVowelsRigidBodies, countablesConsonantsRigidBodies, uncountablesRigidBodies, categories);

        EnvObject apple = objectPool.Get("apple");
        apple.gameObject.transform.position = new Vector3(0, 7, 0);
        objectPool.reset();

        List<EnvObject> countablesVowels = new List<EnvObject>();
        List<EnvObject> countablesConsonants = new List<EnvObject>();
        List<EnvObject> uncountables = new List<EnvObject>();

        foreach(Rigidbody rb in countablesVowelsRigidBodies)
        {
            countablesVowels.Add(new EnvObject(rb));
        }

        foreach(Rigidbody rb in countablesConsonantsRigidBodies)
        {
            countablesConsonants.Add(new EnvObject(rb));
        }

        foreach(Rigidbody rb in uncountablesRigidBodies)
        {
            uncountables.Add(new EnvObject(rb));
        }


        // create the scene
        // Helper.CreatePlane(100,100);

        double containerOffset = 0.2; 
        

        Rigidbody[] allTables = Resources.LoadAll<Rigidbody>($"{Constants.prefabFolder}/tables");
        Rigidbody[] allContainers = Resources.LoadAll<Rigidbody>($"{Constants.prefabFolder}/containers");
        Rigidbody[] allCharacters = Resources.LoadAll<Rigidbody>($"{Constants.prefabFolder}/characters");
        Rigidbody[] allCountables = Resources.LoadAll<Rigidbody>($"{Constants.prefabFolder}/items/countables");


        // Rigidbody table = allTables[0];
        // Rigidbody containerMain = allContainers[0];
        // Rigidbody containerSecondary = allContainers[0];
        // Rigidbody characterMain = allCharacters[0];
        // Rigidbody characterSecondary = allCharacters[0];

        List<float> containerPositionFactors = new List<float>(){-0.25f, 0, 0.25f};
        List<float> cameraPositionFactors = new List<float>(){-0.25f, 0, 0.25f};
        List<Scene> scenes = new List<Scene>();

        Rigidbody characterMainInstance, characterSecondaryInstance, tableInstance, containerMainInstance, containerSecondaryInstance;
        Bounds tableBounds, containerMainBounds, containerSecondaryBounds, characterBounds;
        EnvObject characterMainEnvObject, characterSecondaryEnvObject, tableEnvObject, containerMainEnvObject, containerSecondaryEnvObject;
        Vector3 cameraPosition, cameraLook, tablePos; 
        Dictionary<string, EnvObject> sceneObjects;
        float secTrayOffset = 0f; 
        float mainTrayOffset = 0f;
        Vector3 secTrayRotation = new Vector3(0, 180f, 0);
        Vector3 mainTrayRotation = new Vector3(0, 0, 0);

        foreach (Rigidbody character in allCharacters)
        {
            foreach (Rigidbody table in allTables)
            {
                foreach (Rigidbody container in allContainers)
                {
                    foreach (float containerPositionFactor in containerPositionFactors)
                    {
                        foreach (float cameraPositionFactor in cameraPositionFactors)
                        {
                            tableBounds = Helper.getObjectBounds(table);
                            containerMainBounds = Helper.getObjectBounds(container);
                            containerSecondaryBounds = Helper.getObjectBounds(container);
                            characterBounds = Helper.getObjectBounds(character);

                            if(table.name == "circularTable" && containerPositionFactor !=0)
                            {
                                secTrayOffset = 1.5f;
                                // secTrayRotation = new Vector3(0, 180f + Math.Sign(containerPositionFactor) * 25, 0f);
                            }else {
                                secTrayOffset = 0f;
                                secTrayRotation = new Vector3(0, 180f, 0);
                            }

                            if(table.name == "circularTable" && cameraPositionFactor !=0)
                            {
                                mainTrayOffset = 1.5f;
                                // mainTrayRotation = new Vector3(0, 0f - Math.Sign(cameraPositionFactor) * 25, 0);
                            }else {
                                mainTrayOffset = 0f;
                                mainTrayRotation = new Vector3(0, 0, 0);
                            }
                            

                            characterMainInstance = Instantiate(character, new Vector3(tableBounds.size[0] * cameraPositionFactor, 0f, -tableBounds.size[2]/2 - characterBounds.size[2]/2 + mainTrayOffset), Quaternion.Euler(mainTrayRotation.x, mainTrayRotation.y, mainTrayRotation.z));
                            characterSecondaryInstance = Instantiate(character, new Vector3(tableBounds.size[0] * containerPositionFactor, 0f, tableBounds.size[2]/2 + characterBounds.size[2]/2 - secTrayOffset), Quaternion.Euler(secTrayRotation.x, secTrayRotation.y, secTrayRotation.z));
                            containerMainInstance = Instantiate(container, new Vector3(tableBounds.size[0] * cameraPositionFactor, tableBounds.size[1] + containerMainBounds.size[1]/2, -tableBounds.size[2]/2 + (float) containerOffset + containerSecondaryBounds.size[2]/2 + mainTrayOffset ),  Quaternion.Euler(mainTrayRotation.x, mainTrayRotation.y, mainTrayRotation.z) * allContainers[0].transform.localRotation);
                            containerSecondaryInstance = Instantiate(container, new Vector3(tableBounds.size[0] * containerPositionFactor, tableBounds.size[1] + containerMainBounds.size[1]/2, tableBounds.size[2]/2 - (float) containerOffset - containerMainBounds.size[2]/2 - secTrayOffset), Quaternion.Euler(secTrayRotation.x, secTrayRotation.y, secTrayRotation.z) * allContainers[0].transform.localRotation);
                            tableInstance = Instantiate(table, new Vector3(0f, tableBounds.size[1]/2, 0f), Quaternion.identity);

                            cameraPosition = new Vector3(characterMainInstance.gameObject.transform.position.x, characterMainInstance.gameObject.transform.position.y + characterBounds.size[1]-1, characterMainInstance.gameObject.transform.position.z+1);
                            cameraLook = new Vector3(table.gameObject.transform.position.x, table.gameObject.transform.position.y + tableBounds.size[1]/2 +2, table.gameObject.transform.position.z + 2);

                            characterMainInstance.gameObject.SetActive(false);
                            characterSecondaryInstance.gameObject.SetActive(false);
                            containerMainInstance.gameObject.SetActive(false);
                            containerSecondaryInstance.gameObject.SetActive(false);
                            tableInstance.gameObject.SetActive(false);

                            sceneObjects = new Dictionary<string, EnvObject>();
                            sceneObjects.Add("characterMain", new EnvObject(characterMainInstance));
                            sceneObjects.Add("characterSecondary", new EnvObject(characterSecondaryInstance));
                            sceneObjects.Add("containerMain", new EnvObject(containerMainInstance));
                            sceneObjects.Add("containerSecondary", new EnvObject(containerSecondaryInstance));
                            sceneObjects.Add("table", new EnvObject(tableInstance));
                            
                            Scene scene = new Scene(
                                sceneObjects,
                                cameraPosition,
                                cameraLook
                            );

                            scenes.Add(scene);
                        }
                    }
                }
            }
        }

        
        // Rigidbody tableOne = allTables[0];
        // Rigidbody containerMain = allContainers[0];
        // Rigidbody containerSecondary = allContainers[0];
        // Rigidbody characterMain = allCharacters[0];
        // Rigidbody characterSecondary = allCharacters[0];

        // // tablePrefab = Resources.load("Assets/prefabs/Table.prefab") as Rigidbody;
        // tableBounds = Helper.getObjectBounds(tableOne);
        // containerMainBounds = Helper.getObjectBounds(allContainers[0]);
        // containerSecondaryBounds = Helper.getObjectBounds(allContainers[0]);
        // characterBounds = Helper.getObjectBounds(allCharacters[0]);

        // characterMainInstance = Instantiate(characterMain, new Vector3(0f, 0f, -tableBounds.size[2]/2 - characterBounds.size[2]/2), Quaternion.identity);
        // characterSecondaryInstance = Instantiate(characterSecondary, new Vector3(0f, 0f, tableBounds.size[2]/2 + characterBounds.size[2]/2), Quaternion.Euler(0f, 180f, 0f));
        // containerMainInstance = Instantiate(containerMain, new Vector3(0f, tableBounds.size[1] + containerMainBounds.size[1]/2, -tableBounds.size[2]/2 + (float) containerOffset + containerSecondaryBounds.size[2]/2 ), Quaternion.identity * allContainers[0].transform.localRotation);
        // containerSecondaryInstance = Instantiate(containerSecondary, new Vector3(0f, tableBounds.size[1] + containerMainBounds.size[1]/2, tableBounds.size[2]/2 - (float) containerOffset - containerMainBounds.size[2]/2), Quaternion.identity * allContainers[0].transform.localRotation);
        // tableInstance = Instantiate(tableOne, new Vector3(0f, tableBounds.size[1]/2, 0f), Quaternion.identity);
        // cameraPosition = new Vector3(characterMainInstance.gameObject.transform.position.x, characterMainInstance.gameObject.transform.position.y + characterBounds.size[1]-1, characterMainInstance.gameObject.transform.position.z+1);
        // // Camera.main.transform.position = new Vector3(characterMainInstance.gameObject.transform.position.x, characterMainInstance.gameObject.transform.position.y + characterBounds.size[1]-1, characterMainInstance.gameObject.transform.position.z+1);
        // tablePos = tableInstance.gameObject.transform.position; 
        // cameraLook = new Vector3(tablePos.x, tablePos.y + tableBounds.size[1]/2, tablePos.z + 2);
        // // Camera.main.transform.LookAt(new Vector3(tablePos.x, tablePos.y + tableBounds.size[1]/2, tablePos.z + 2));

        // tableEnvObject = new EnvObject(tableInstance);
        // containerMainEnvObject = new EnvObject(containerMainInstance);
        // containerSecondaryEnvObject = new EnvObject(containerSecondaryInstance);
        // characterMainEnvObject = new EnvObject(characterMainInstance);
        // characterSecondaryEnvObject = new EnvObject(characterSecondaryInstance);
    
        // Scene scene = new Scene(tableEnvObject, containerMainEnvObject, containerSecondaryEnvObject, characterMainEnvObject, characterSecondaryEnvObject, cc_inputs, cc_outputs, categories, cameraPosition, cameraLook);
        // // Scene[] scenes = {scene};
        // scenes.Add(scene);

        // List<string> determiners = new List<string> {"my"};

        DeterminerGenerator determinerGenerator = new DeterminerGenerator(objectPool, countableQuantities, uncountableQuantities, cc_inputs, cc_outputs, categories);
        List<string> determiners = new List<string> {"a", "an", "all", "any", "every", "my", "your", "this", "that", "these", "those", "some", "many", "few", "both", "neither", "little", "much", "either", "our", "no", "the", "half", "several", "each"};

        await Task.Delay(400);
        foreach(string determiner in determiners)
        {  
            Debug.Log("test");
            await determinerGenerator.generate(10000, determiner, scenes);
        };

        // await DeterminerGenerator.generate("that", cc_inputs, cc_outputs,  tableInstance, containerMainInstance, containerSecondaryInstance, characterMainInstance, characterSecondaryInstance);

        Helper.saveCOCO(determinerGenerator.cc_inputs, $"{Application.streamingAssetsPath}/{Constants.datasetFolder}/{Constants.annotationsFolder}", "annotations_full.json");
        // Helper.saveCOCO(determinerGenerator.cc_outputs, $"{Application.streamingAssetsPath}/{Constants.datasetFolder}/{Constants.annotationsFolder}", "train_output_labels.json");
        EditorApplication.isPlaying = false;
    }

    // Update is called once per frame
    void Update()
    {
        // if Input.GetKeyDown()
        
    }
}
