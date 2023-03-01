using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;

public class Scene: Object
{
    // EnvObject is a custom object made to simplify the extraction of important features of the game object 
    // public EnvObject table; 
    // public EnvObject containerMain; 
    // public EnvObject containerSecondary;
    // public EnvObject characterMain;
    // public EnvObject characterSecondary;
    public Dictionary<string, EnvObject> objects;
    public COCOAnnotations cc_inputs;
    public COCOAnnotations cc_outputs;
    public Dictionary<string, Category> categories;
    public Vector3 cameraPosition; 
    public Vector3 cameraLook; 

    public Scene(Dictionary<string, EnvObject> objects, Vector3 cameraPosition, Vector3 cameraLook)
    {
        // this.table = table;
        // this.containerMain = containerMain;
        // this.containerSecondary = containerSecondary;
        // this.characterMain = characterMain;
        // this.characterSecondary = characterSecondary;
        this.objects = objects;
        EnvObject table = objects["table"];
        EnvObject containerMain = objects["containerMain"];
        EnvObject containerSecondary = objects["containerSecondary"];
        EnvObject characterMaiecn = objects["characterMain"];
        EnvObject characterSecondary = objects["characterSecondary"];

        this.cameraPosition = cameraPosition;
        this.cameraLook = cameraLook;
    }

    public void SetActive() 
    {
        foreach(EnvObject obj in objects.Values)
        {
            obj.gameObject.SetActive(true);
        }
        Camera.main.transform.position = cameraPosition;
        Camera.main.transform.LookAt(cameraLook);
    }

    public void SetInactive() 
    {
        foreach(EnvObject obj in objects.Values)
        {
            obj.gameObject.SetActive(false);
        }
    }
}

public class SceneConfig: Object 
{ 
    public List<int> totalObjCounts;
    public string mainObjSet;
    public List<int> numSelectedRange;
    public int nExtra;
    public bool useQuantityConfig;
    public Spawn selectedMainObjSpawns;
    public Spawn unselectedMainObjSpawns;

    public SceneConfig(List<int> totalObjCounts, string mainObjSet, List<int> numSelectedRange, int nExtra, bool useQuantityConfig, Spawn selectedMainObjSpawns, Spawn unselectedMainObjSpawns)
    {
        this.totalObjCounts = totalObjCounts;
        this.mainObjSet = mainObjSet;
        this.numSelectedRange = numSelectedRange;
        this.nExtra = nExtra;
        this.useQuantityConfig = useQuantityConfig;
        this.selectedMainObjSpawns = selectedMainObjSpawns;
        this.unselectedMainObjSpawns = unselectedMainObjSpawns;
    }
}

public class Assets: Object 
{ 
    public Dictionary<string, Dictionary<string, List<int>>>countableQuantities;
    public Dictionary<string, Dictionary<string, List<float>>> uncountableQuantities;
    public COCOAnnotations cc_inputs;
    public COCOAnnotations cc_outputs;
    public Dictionary<string, Category> categories;


    public Assets(Dictionary<string, Dictionary<string, List<int>>> countableQuantities, Dictionary<string, Dictionary<string, List<float>>>uncountableQuantities, COCOAnnotations cc_inputs, COCOAnnotations cc_outputs, Dictionary<string, Category> categories)
    {
        this.countableQuantities = countableQuantities;
        this.uncountableQuantities = uncountableQuantities;
        this.cc_inputs = cc_inputs;
        this.cc_outputs = cc_outputs;
        this.categories = categories;
    }
}