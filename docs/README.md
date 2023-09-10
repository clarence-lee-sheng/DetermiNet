
<p align="center" style="font-size:30px; font-weight:bold"> 
DetermiNet: A Large-Scale Diagnostic Dataset for Complex Visually-Grounded Referencing using Determiners
</p>

<div align="center" style="display:flex; justify-content: space-evenly">
   <div>Clarence Lee<sup>1</sup></div>
   <div>M Ganesh Kumar<sup>2</sup></div>
   <div>Cheston Tan<sup>2</sup></div>
</div>

<div align="center" style="display:flex; justify-content:center">
   <div style="margin-right:20px">Design and Artificial Intelligence, SUTD<sup>1</sup></div>
   <div style="margin-left:20px">Centre for Frontier AI Research, A*STAR<sup>2</sup></div>
   <!-- <div>m_ganeshkumar@u.nus.edu</div>
   <div>cheston-tan@i2r.a-star.edu.sg</div> -->
</div>



<br> 

Determiners are an important word class that is used in the referencing and quantification of nouns. However existing datasets place less emphasis on determiners, compared to other word classes. Hence, we have designed the DetermiNet dataset, which is a visuolinguistic dataset comprising of the word class determiners. It comprises of 25 determiners with 10,000 examples each, totalling 250,000 samples. All scenes were synthetically generated using unity. The task is to predict bounding boxes to identify objects of interest, constrained by the semantics of the determiners   

<div align="center">
  <figure>
    <br>
    <img src="./assets/generate_scene.gif" width=1000px/>
  </figure>
</div>

<div align="center" style="font-size:18px; display:flex; justify-content: center">
<a href="https://arxiv.org/abs/2309.03483" style="margin-right:15px; margin-left:-15px">[Paper]</a>
<a href="test" style="margin-right:15px">[Supp]</a>
<a href="https://github.com/clarence-lee-sheng/DetermiNet">[Github]</a>
</div>


### Download the Dataset 
You may download DetermiNet comprising of 250,000 image caption pairs here: 
[https://drive.google.com/drive/folders/1J5dleNxWvFUip5RBsTl6OqQBtpWO0r1k?usp=sharing](https://drive.google.com/drive/folders/1J5dleNxWvFUip5RBsTl6OqQBtpWO0r1k?usp=sharing )

### Real Dataset 
Download real dataset comprising of 100 image caption pairs here: [https://drive.google.com/drive/folders/1J5dleNxWvFUip5RBsTl6OqQBtpWO0r1k?usp=sharing](https://drive.google.com/drive/folders/1J5dleNxWvFUip5RBsTl6OqQBtpWO0r1k?usp=sharing )


### Citation Information
```
@misc{lee2023determinet,
      title={DetermiNet: A Large-Scale Diagnostic Dataset for Complex Visually-Grounded Referencing using Determiners}, 
      author={Clarence Lee and M Ganesh Kumar and Cheston Tan},
      year={2023},
      eprint={2309.03483},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Evaluation scripts 
DetermiNet uses a modified ground truth for multiple annotations, to run your models for evaluation, refer to our [github](https://github.com/clarence-lee-sheng/DetermiNet)


### DetermiNet examples (25 Determiners, 4 Determiner classes)

<div align="center">
  <figure>
    <div>
    <h4>Articles</h4>
    <img src="./assets/determiner_examples/articles.png" width=1000px/>
    <h4>Possessives</h4>
    <img src="./assets/determiner_examples/possessives.png" width=1000px/>
    <h4>Demonstratives</h4>
    <img src="./assets/determiner_examples/demonstratives.png" width=1300px/>
    <h4>Quantifiers</h4>
    <img src="./assets/determiner_examples/quantifiers.png" width=1000px/>
    </div>
  </figure>
</div>

### Contact us
- Clarence Lee: [clarence_leesheng@mymail.sutd.edu.sg]()
- M Ganesh Kumar: [m_ganeshkumar@u.nus.edu]()
- Cheston Tan: [cheston-tan@i2r.a-star.edu.sg]()





<!-- If the dataset has a [DOI](https://www.doi.org/), please provide it here. -->