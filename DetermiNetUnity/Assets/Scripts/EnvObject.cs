using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;

public class EnvObject : Object
{
    public GameObject gameObject { get; set; }
    public Rigidbody rigidBody {get; set;}
    public float x {get; set;}
    public float y {get; set;}
    public float z {get; set;}
    public float width {get; set;}
    public float height {get; set;}
    public float depth {get; set;}
    public string name {get; set;}
    public float liquidLevel {get; set;}
    // public Rect boundingBoxRect {get; set;}

    public EnvObject(Rigidbody body, float liqLevel=-1)
    {
        rigidBody = body;
        gameObject = rigidBody.gameObject;
        name = gameObject.name.Replace("(Clone)", "");

        Bounds objBounds = Helper.getObjectBounds(body);

        x = (float) body.transform.position[0];
        y = (float) body.transform.position[1];
        z = (float) body.transform.position[2];

        width = (float) objBounds.size[0];
        height = (float) objBounds.size[1];
        depth = (float) objBounds.size[2];

        liquidLevel = liqLevel; 
        // boundingBoxRect = getGUIBoundingBox(body);
    }

    // public Rigidbody spawnRandomlyOnObjects(EnvObject[] objs, Character character)
    // {
    //     int offset = 1;
    //     float x = Random.Range(obj.x - obj.width/2 + width/2, obj.x + obj.width/2 - width/2);
    //     float z = Random.Range(obj.z - obj.depth/2 + depth/2, obj.z + obj.depth/2 - depth/2);
    //     int counter = 0; // prevent infinite loop
    //     Vector3 spawnPoint = new Vector3(x, obj.y + height/2 + obj.height/2 + 0.01f, z);
    //     // Debug.Log($"Object y position: {obj.y}");
    //     Vector3 viewportPoint = Camera.main.WorldToViewportPoint(spawnPoint);
    //     bool isInCamera = (new Rect(0, 0, 1, 1)).Contains(viewportPoint);
    //     bool isValidIntersectWithCircle = (new Vector2(x, z) - new Vector2(character.x, character.z)).magnitude < radius == character.within;
    //     bool inTable; 
    //     Debug.Log($"Spawn Obj Name: {obj.name}");

    //     if (obj.name == "circularTable")
    //     {
    //         inTable = (new Vector2(x, z) - new Vector2(obj.x, obj.z)).magnitude < obj.width/2;
    //     }else{
    //         inTable = true;
    //     }
    //     // Debug.Log($"center_x: {center_x}, center_z: {center_z}, x: {x}, z: {z}, radius: {radius}, inCircle: {isValidIntersectWithCircle}");
    //     while(!inTable || isColliding(spawnPoint) || !isInCamera || !isValidIntersectWithCircle)
    //     {
    //         // Debug.Log($"isColliding: {isColliding(spawnPoint)}, isInCamera: {isInCamera}, isValidIntersectWithCircle: {isValidIntersectWithCircle}");
    //         x = Random.Range(obj.x - obj.width/2 + width/2, obj.x + obj.width/2 - width/2);
    //         z = Random.Range(obj.z - obj.depth/2 + depth/2, obj.z + obj.depth/2 - depth/2);
    //         spawnPoint = new Vector3(x, obj.y + height/2 + obj.height/2 + 0.01f, z);
    //         viewportPoint = Camera.main.WorldToViewportPoint(spawnPoint);
    //         isInCamera = (new Rect(0, 0, 1, 1)).Contains(viewportPoint);
    //         isValidIntersectWithCircle = (new Vector2(x, z) - new Vector2(character.x, character.z)).magnitude < radius == character.within;
    //         if (obj.name.Replace("(Clone)", "") == "circularTable")
    //         {
    //             inTable = (new Vector2(x, z) - new Vector2(obj.x, obj.z)).magnitude < obj.width/2;
    //         }else{
    //             inTable = true;
    //         }
            
    //         if (counter > 1000)
    //         {
    //             Debug.Log("Infinite loop detected");
    //             break;
    //         }
    //         counter++;
    //     }
    //     // Debug.Log($"Spawn point: {spawnPoint}");
    //     Rigidbody body = Instantiate(rigidBody, spawnPoint, Quaternion.identity);
    //     return body;
    // }

    public bool isColliding(Vector3 spawnPoint)
    {
        return Physics.CheckBox(spawnPoint, new Vector3(width/2, height/2, depth/2));
    } 
}   



