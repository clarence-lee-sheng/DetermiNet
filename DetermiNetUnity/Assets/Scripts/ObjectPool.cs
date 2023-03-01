using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;
// using System;

public class ObjectPool : MonoBehaviour
{
    public Dictionary<string, EnvObject> objects; 
    public Dictionary<string, List<EnvObject>> pools; 
    public Dictionary<string, List<EnvObject>> active; 
    public Dictionary<string, List<string>> objectSets;
    public Dictionary<string, float> uncountableLiquidScales;
    public Dictionary<string, Category> categories;
    public Dictionary<string, int> containerCounts;
    private int maxContainerCount = 7;


    public ObjectPool(Rigidbody[] countablesVowels, Rigidbody[] countablesConsonants, Rigidbody[] uncountables, Dictionary<string, Category> categories)
    {
        this.objects = new Dictionary<string, EnvObject>();
        this.pools = new Dictionary<string, List<EnvObject>>();
        this.active = new Dictionary<string, List<EnvObject>>();
        this.objectSets = new Dictionary<string, List<string>>();
        this.uncountableLiquidScales = new Dictionary<string, float>();
        this.categories = categories;
        this.containerCounts = new Dictionary<string, int>();

        containerCounts.Add("containerMain", 0);
        containerCounts.Add("containerSecondary", 0);
        containerCounts.Add("table", 0);

        objectSets.Add("countablesVowels", new List<string>());
        objectSets.Add("countablesConsonants", new List<string>());
        objectSets.Add("uncountables", new List<string>());
        objectSets.Add("all", new List<string>());
        objectSets.Add("countables", new List<string>());

        foreach (Rigidbody obj in countablesVowels)
        {
            this.objects[obj.gameObject.name] = new EnvObject(obj);
            this.pools[obj.gameObject.name] = new List<EnvObject>();
            this.objectSets["countablesVowels"].Add(obj.gameObject.name);
            this.objectSets["all"].Add(obj.gameObject.name);
            this.objectSets["countables"].Add(obj.gameObject.name);
        }

        foreach (Rigidbody obj in countablesConsonants)
        {
            this.objects[obj.gameObject.name] = new EnvObject(obj);
            this.pools[obj.gameObject.name] = new List<EnvObject>();
            this.objectSets["countablesConsonants"].Add(obj.gameObject.name);
            this.objectSets["all"].Add(obj.gameObject.name);
            this.objectSets["countables"].Add(obj.gameObject.name);
        }

        foreach (Rigidbody obj in uncountables)
        {
            this.objects[obj.gameObject.name] = new EnvObject(obj);
            this.pools[obj.gameObject.name] = new List<EnvObject>();
            this.objectSets["uncountables"].Add(obj.gameObject.name);
            this.objectSets["all"].Add(obj.gameObject.name);
            this.uncountableLiquidScales.Add(obj.gameObject.name, obj.gameObject.transform.GetChild(1).transform.localScale[2]);
        }

        this.active.Add("target", new List<EnvObject>());
        this.active.Add("notTarget", new List<EnvObject>());
    }

    public EnvObject Get(string name, string type="notTarget")
    {
        EnvObject newObj; 
        List<EnvObject> pool = this.pools[name];
        int lastIndex = pool.Count - 1;

        if(lastIndex > 0)
        { 
            newObj = pool[lastIndex];
            this.pools[name].RemoveAt(lastIndex);
        }
        else 
        {
            Rigidbody newSpawn = Instantiate(this.objects[name].rigidBody); 
            newObj = new EnvObject(newSpawn); 
        }
        newObj.gameObject.SetActive(true);
        active[type].Add(newObj); 
        return newObj; 
    }


    public void spawnInScene(int numObjects, List<string> names, List<EnvObject> objs, Character character, string type, List<float> liquidRange, bool includeLiquidRange = true)
    {
        EnvObject itemToSpawn;
        string name;
        float liquidScale;
        Vector3 scale; 
        float buffer = 0.1f;
        for (int i=0; i < numObjects; i++)
        {
            name = Helper.getRandomFromList(names);
            itemToSpawn = Get(name, type);
            if (categories[name].supercategory.Split("_")[0] == "uncountable")
            {
                if (categories[name].supercategory.Split("_")[1]== "liquid")
                {
                    if (includeLiquidRange)
                    {
                        liquidScale = Random.Range(liquidRange[0], liquidRange[1]);
                        scale = itemToSpawn.gameObject.transform.GetChild(1).transform.localScale;
                        scale[2] = liquidScale * uncountableLiquidScales[name];
                        itemToSpawn.gameObject.transform.GetChild(1).transform.localScale = scale;
                    }
                    else
                    {
                        liquidScale = Random.Range(0.1f, 1f);
                        while (liquidScale > liquidRange[0] - buffer && liquidScale < liquidRange[1] + buffer)
                        {
                            liquidScale = Random.Range(0.1f, 1f);
                        }
                        scale = itemToSpawn.gameObject.transform.GetChild(1).transform.localScale;
                        scale[2] = liquidScale * uncountableLiquidScales[name];
                        itemToSpawn.gameObject.transform.GetChild(1).transform.localScale = scale;
                    }
                    itemToSpawn.liquidLevel = liquidScale;
                    // itemToSpawn.rigidBody.gameObject.GetChild(1).transform.localScale[2] = Helper.getRandomFromRange(liquidRange) * this.uncountableLiquidScales[name];
                }
            }
                
            
            EnvObject obj = objs[UnityEngine.Random.Range(0, objs.Count)];
            // Debug.Log(obj.name); 
            // Debug.Log(obj.gameObject.transform.position);
            // Debug.Log($"Object wdith: {obj.width}, Object height: {obj.depth}");   
            // Debug.Log($"{Helper.getObjectBounds(obj.rigidBody)}");
            int offset = 1;
            float x = Random.Range(obj.x - obj.width/2 + itemToSpawn.width/2 + 0f, obj.x + obj.width/2 - itemToSpawn.width/2 - 0f);
            float z = Random.Range(obj.z - obj.depth/2 + itemToSpawn.depth/2 + 0f, obj.z + obj.depth/2 - itemToSpawn.depth/2 - 0f);
            int counter = 0; // prevent infinite loop
            int iter = 0;
            Vector3 spawnPoint = new Vector3(x, obj.y + itemToSpawn.height/2 + obj.height/2 + 0.001f, z);
            Vector3 viewportPoint = Camera.main.WorldToViewportPoint(spawnPoint);
            bool isInCamera = (new Rect(0, 0, 1, 1)).Contains(viewportPoint);
            bool isValidIntersectWithCircle = (new Vector2(x, z) - new Vector2(character.x, character.z)).magnitude < character.reach == character.within;
            bool inTable; 

            

            if (obj.name == "circularTable")
            {
                inTable = (new Vector2(x, z) - new Vector2(obj.x, obj.z)).magnitude < (obj.width/2 - 2);
            }else{
                inTable = true;
            }
            // Debug.Log($"x: {x}, z: {z}");
            while(!inTable || itemToSpawn.isColliding(spawnPoint) || !isInCamera || !isValidIntersectWithCircle)
            {
                
                x = Random.Range(obj.x - obj.width/2 + itemToSpawn.width/2 + 0f, obj.x + obj.width/2 - itemToSpawn.width/2 - 0f);
                z = Random.Range(obj.z - obj.depth/2 + itemToSpawn.depth/2 + 0f, obj.z + obj.depth/2 - itemToSpawn.depth/2 - 0f);
                spawnPoint = new Vector3(x, obj.y + itemToSpawn.height/2 + obj.height/2 + 0.001f, z);
                viewportPoint = Camera.main.WorldToViewportPoint(spawnPoint);
                isInCamera = (new Rect(0, 0, 1, 1)).Contains(viewportPoint);
                isValidIntersectWithCircle = (new Vector2(x, z) - new Vector2(character.x, character.z)).magnitude < character.reach == character.within;
                if (obj.name == "circularTable")
                { 
                    inTable = (new Vector2(x, z) - new Vector2(obj.x, obj.z)).magnitude <  (obj.width/2 - 2);
                    // Debug.Log($"In table {inTable}");
                }else{
                    inTable = true;
                }
                
                if (counter > 1000)
                {
                    counter = 0;
                    iter += 1; 
                    obj = objs[UnityEngine.Random.Range(0, objs.Count)];
                    // break;
                }
                counter++;
                if (iter > 20)
                {
                    Debug.Log($"inTable: {inTable}, isColliding: {itemToSpawn.isColliding(spawnPoint)}, isInCamera: {isInCamera}, isValidIntersectWithCircle: {isValidIntersectWithCircle}");
                    Debug.Log("Infinite loop");
                    break;
                }
            }
            itemToSpawn.gameObject.transform.position = spawnPoint;
            // Debug.Log($"Limits: {obj.x - obj.width/2 + itemToSpawn.width/2 +0}, {obj.x + obj.width/2 - itemToSpawn.width/2 }, {obj.z - obj.depth/2 + itemToSpawn.depth/2 +0}, {obj.z + obj.depth/2 - itemToSpawn.depth/2 }");
            // Debug.Log($"Spawned {itemToSpawn.name} at {spawnPoint}");
            // Debug.Log($"{spawnPoint.x}, {spawnPoint.z}");
        }
        
    }


    public void reset()
    {
        foreach (KeyValuePair<string, List<EnvObject>> activeObjects in this.active)
        {
            foreach (EnvObject obj in activeObjects.Value)
            {
                obj.gameObject.SetActive(false);
                obj.gameObject.layer = 0;

                foreach (Transform child in obj.gameObject.transform)
                {
                    child.gameObject.layer = 0;
                }
                this.pools[obj.name].Add(obj);
                
            }
            
            activeObjects.Value.Clear();
        }
    }
}

