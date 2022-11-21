using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

public class Helper : MonoBehaviour
{
    public static Rigidbody CreatePlane(float width, float height)
    {
        GameObject plane = new GameObject("Plane"); 
        MeshFilter meshFilter = plane.AddComponent<MeshFilter>();
        MeshRenderer meshRenderer = plane.AddComponent<MeshRenderer>();
        BoxCollider boxCollider = plane.AddComponent<BoxCollider>();
        boxCollider.size = new Vector3(width, 0f, height);

        Rigidbody rigidPlane = plane.AddComponent<Rigidbody>();
        rigidPlane.useGravity=false;
        rigidPlane.isKinematic=true;
        Mesh mesh = new Mesh();
        mesh.vertices = new Vector3[] {
            new Vector3(-width/2, 0, -height/2),
            new Vector3(-width/2, 0, height/2),
            new Vector3(width/2, 0, height/2),
            new Vector3(width/2, 0, -height/2)
        };
        mesh.triangles = new int[] {0, 1, 2, 0, 2, 3};
        mesh.uv = new Vector2[] {
            new Vector2(0, 0),
            new Vector2(0, 1),
            new Vector2(1, 1),
            new Vector2(1, 0)
        };
        meshFilter.mesh = mesh;
        return rigidPlane;
    }

    public static Bounds getObjectBounds<T>(T obj) where T : Component
    {
        Bounds bounds = obj.GetComponent<MeshRenderer>().bounds; 
        MeshRenderer[] meshes = obj.GetComponentsInChildren<MeshRenderer>();
        SkinnedMeshRenderer[] skinnedMeshes = obj.GetComponentsInChildren<SkinnedMeshRenderer>();
        foreach (MeshRenderer mesh in meshes )
        {
            bounds.Encapsulate(mesh.bounds);
        }

        foreach (SkinnedMeshRenderer skinnedMesh in skinnedMeshes )
        {
            bounds.Encapsulate(skinnedMesh.bounds);
        }

        return bounds;
    }

    public static T getRandomFromList<T>(T[] list, int idxToExclude = 100000 )
    {
        int idx = idxToExclude;
        while (idx == idxToExclude)
        {
            idx= Random.Range(0, list.Length);
        }
        return list[idx];
    }

    public static T getRandomFromList<T>(List<T> list, int idxToExclude = 100000 )
    {
        int idx = idxToExclude;
        while (idx == idxToExclude)
        {
            idx= Random.Range(0, list.Count);
        }
        return list[idx];
    }


    public static Rect getGUIBoundingBoxRect(Rigidbody rb, Camera camera)
    {
        GameObject go = rb.gameObject;

        // find bounding boxes based on mesh
        Vector3[] vertices = go.GetComponent<MeshFilter>().mesh.vertices;

        // apply the world transforms (position, rotation, scale) to the mesh points and then get their 2D position
        // relative to the camera
        Vector2[] vertices_2d = new Vector2[vertices.Length];
        for (var i = 0; i < vertices.Length; i++)
        {
            vertices_2d[i] = camera.WorldToScreenPoint(go.transform.TransformPoint( vertices[i]));
        }

        // find the min max bounds of the 2D points
        Vector2 min = vertices_2d[0];
        Vector2 max = vertices_2d[0];
        foreach (Vector2 vertex in vertices_2d)
        {
            min = Vector2.Min(min, vertex);
            max = Vector2.Max(max, vertex);
        }

        Rect boundingBox = new Rect(Mathf.Max(min.x, 0), Mathf.Max(Constants.imgHeight - max.y,0), Mathf.Min(max.x-Mathf.Max(min.x, 0), Constants.imgWidth-min.x),Mathf.Min(max.y-min.y, Constants.imgHeight-min.y));
        
        return boundingBox;
    }

    public static List<int> getBoundingBox(Rigidbody rb, Camera camera)
    {
        Rect rect = Helper.getGUIBoundingBoxRect(rb, camera);
        int x = (int)rect.x;
        int y = (int)rect.y;
        int width = (int)rect.width;
        int height = (int)rect.height;
        return new List<int> {x, y, width, height};
    }

    public static void saveCOCO(COCOAnnotations cocoAnnotations, string filepath, string filename)
    {
        string jsondata = JsonUtility.ToJson(cocoAnnotations);
        System.IO.File.WriteAllText(filepath + $"/{filename}", jsondata);
    }

    public static void synthTakeSnapshot(int channel,string filepath,string filename)
    {
        ImageSynthesis synth;
        synth = Camera.main.GetComponent<ImageSynthesis>();
        synth.OnSceneChange();
        synth.Save($"{filename}.png", Constants.imgWidth, Constants.imgHeight, filepath, channel);
    }
}
