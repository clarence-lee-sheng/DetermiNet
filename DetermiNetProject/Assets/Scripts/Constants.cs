using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Constants : MonoBehaviour
{
    
    public static string prefabFolder = "prefabs";
    public static string datasetFolder = "dataset";
    public static string annotationsFolder = "annotations";
    public static string countablesVowelsPath = $"{prefabFolder}/items/countables/vowels";
    public static string countablesConsonantsPath = $"{prefabFolder}/items/countables/consonants";
    public static string uncountablesPath = $"{prefabFolder}/items/uncountables";
    public const int imgWidth = 256; 
    public const int imgHeight = 256;
}
