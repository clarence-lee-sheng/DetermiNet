using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;

public class Spawn
{
    public List<string> spawns;
    public CharacterConfig characterConfig; 

    public Spawn(List<string> spawns, CharacterConfig characterConfig)
    {
        this.spawns = spawns; 
        this.characterConfig = characterConfig;
    }

    public Spawn(string state)
    {
        if (state == "all")
        {
            this.spawns = new List<string>{"containerMain", "containerSecondary", "table"};
            this.characterConfig = new CharacterConfig(false);
        }else if(state == "none")
        {
            this.spawns = new List<string>();
            this.characterConfig = new CharacterConfig(false);
        }
    }

}

public class CharacterConfig
{
    public string name; 
    public float reach; 
    public bool within; 

    public CharacterConfig(string name, float reach, bool within)
    {
        this.name = name; 
        this.reach = reach;
        this.within = within; 
    }

    public CharacterConfig(bool isValid = false)
    {
        this.name = "characterMain";
        this.reach = Mathf.Infinity;
        this.within = true; 
    }
}

public class Character 
{
    public float x; 
    public float z; 
    public string name;
    public float reach;
    public bool within;

    public Character(EnvObject charObj, CharacterConfig config)
    {
        this.x = charObj.x;
        this.z = charObj.z;
        this.name = config.name;
        this.reach = config.reach;
        this.within = config.within;
    }
}

