using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

// https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch#:~:text=Five%20COCO%20Annotation%20Types&text=COCO%20has%20five%20annotation%20types,annotations%20are%20stored%20using%20JSON.

public class CountableQuantity
{
    public List<int> few;
    public List<int> some;
    public List<int> many;
    public List<int> several;
    
    public CountableQuantity(List<int> few, List<int> some, List<int> many, List<int> several)
    {
        this.few = few;
        this.some = some;
        this.many = many;
        this.several = several;
    }
}

public class UncountableQuantity
{
    public List<float> little;
    public List<float> some;
    public List<float> many;
    
    public UncountableQuantity(List<float> little, List<float> some, List<float> many)
    {
        this.little = little;
        this.some = some;
        this.many = many;
    }
}


