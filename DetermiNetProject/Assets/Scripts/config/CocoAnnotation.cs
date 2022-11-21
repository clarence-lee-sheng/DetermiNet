using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

// https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch#:~:text=Five%20COCO%20Annotation%20Types&text=COCO%20has%20five%20annotation%20types,annotations%20are%20stored%20using%20JSON.

[Serializable]
public class COCOAnnotations{
    public List<Category> categories = new List<Category>();
    public List<Image> images = new List<Image>();
    public List<SegmentationAnnotation> annotations = new List<SegmentationAnnotation>();
    public List<PhraseAnnotation> phrase_annotations = new List<PhraseAnnotation>();
}

[Serializable]
public class Category
{
    [SerializeField] public int id; 
    [SerializeField] public string name;
    [SerializeField] public string supercategory; 
    
    public Category(string supercategory, int id, string name)
    {
        this.supercategory = supercategory;
        this.id = id;
        this.name = name;
    }
}

[Serializable]
public class Image
{
    [SerializeField] public int id;
    [SerializeField] public string file_name;
    [SerializeField] public int width;
    [SerializeField] public int height;

    public Image(int id, string file_name, int width = Constants.imgWidth, int height = Constants.imgHeight)
    {
        this.id = id;
        this.file_name = file_name;
        this.width = width;
        this.height = height;
    }
}

// [Serializable]
// public class Annotation{ 

// }

[Serializable]
public class SegmentationAnnotation{
    [SerializeField] public int id; 
    [SerializeField] public int image_id;
    [SerializeField] public int category_id;
    [SerializeField] public int area; 
    [SerializeField] public int iscrowd; 
    [SerializeField] public List<int> bbox;
    
    public SegmentationAnnotation(int id, int image_id, int category_id, int area, int iscrowd, List<int> bbox)
    {
        this.id = id;
        this.image_id = image_id;
        this.category_id = category_id;
        this.area = area;
        this.iscrowd = iscrowd;
        this.bbox = bbox;
    }
}

[Serializable]
public class PhraseAnnotation{
    [SerializeField] public int id; 
    [SerializeField] public int image_id;
    [SerializeField] public string caption;
    
    public PhraseAnnotation(int id, int image_id, string caption)
    {
        this.id = id;
        this.image_id = image_id;
        this.caption = caption;
    }
}