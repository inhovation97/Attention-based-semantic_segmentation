# Attention with semantic segmentation


[Publication]    
Inho Jeong, Minyoung Hwang, Chaejun Seo, Seunghyeok Hong. (2023).   
Attention-based Fine-tuning for Reducing Misclassification in Semantic Image Segmentation 한국정보과학회 학술발표논문집,   

------------------------------------------------------------

## **Abstract**  
Semantic segmentation is a field of deep learning research that classifies backgrounds and multiple objects in pixels within an image. This study is conducted for purpose in various fields, and in this paper, the problem of pixel misclassification that occurs when the background and segmentation target are binary divided was analyzed. To improve segmentation accuracy, medical images and satellite image datasets were used to compare and experiment by adding attention modules to existing deep learning algorithm structures that cause misclassification and changing output layer structures. The models used at this time are CNN models 
[FCN(Fully Convolutional Network)](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf), [U-Net](https://arxiv.org/pdf/1505.04597.pdf), [DeepLab V3+](https://arxiv.org/pdf/1802.02611.pdf), which are widely used in semantic segmentation studies, and Transformer-based [FCBFormer](https://arxiv.org/pdf/2208.08352.pdf), which shows high performance in recent medical fields. Finally, we propose a fine tuning method that improves performance compared to the existing method for the four experimented deep learning algorithms. In common with all algorithms, the results of the fine tuning model proposed in this paper were the best, and the performance was improved by up to 2.9% based on Dice Score.   

## **Problems of misclassification in semantic segmentation**   
When used to distinguish lesions and backgrounds in medical images and divide road and non-road areas in satellite images, input image ```X``` is an R<sup>H×W×C</sup> three-dimensional dataset with ```H × W``` number of pixels and expressed as ```C``` according to scale. When input image ```X``` is input to deep learning algorithm ```F```, **P(F(X)) = [0,1] ∈ R<sup>H×W</sup>** is finally obtained and binary classification is performed through the probability value of each pixel.   

### **Figure 1**    
<img src="https://user-images.githubusercontent.com/104747868/235346780-85e285b2-43d6-47f3-b19a-b6a2b4034d83.png" width="600" height="300">    

+ It is a schematic of the endoscopic (Kvasir-SEG) dataset image to distinguish the segmentation target **Gastrointestinal polyp** from the background and the learning results of ```U-net```, which are widely used in the medical field.   
+ The problem of misclassification commonly observed in the deep learning algorithm was defined in two patterns.   
**① Pixels that are misclassified due to lack of information (size, texture, etc.) to be segmented**   
**② Pixels with a segmentation target/background binary classification probability close to 0.5 and misclassified**     

In the case of pattern ①, information on the segmentation target from the algorithm is insufficient. Therefore, it is interpreted that a feature capable of having more diverse information is needed.   

On the other hand, in the case of pattern ②, pixels with a probability value of 0.5 derived from the algorithm as a binary classification, which is the center value in the range of 0-1, were observed to be misclassified well.    

That is, it was intended to induce the area classification to be clear by bringing the inference probability of the segmentation target closer to the value of 0 or 1.   

------------------------------------------

## Improvement of the Misclassification Problem    

In order to reduce the error for the two patterns of the above figure 1, an attempt was made as follows.   

**a. It further strengthens the features obtained from the algorithm.**    
**b. Derive a value close to both extremes so that the probability value does not stay near 0.5.**    

Figure 2 is an example of a structure in which two improvements a and b are targeted.    

### Figure 2   
<img src="https://user-images.githubusercontent.com/104747868/235344119-ae5d61a7-7176-4955-94b3-501c89c44294.png" width="600" height="300">   

[CBAM](https://arxiv.org/pdf/1807.06521.pdf) was added as the attention module (Figure 2a). The CBAM module's **spatial attention map** plays a role in giving the final output map a weight of **0-1** according to the pixel importance while passing through the ```sigmoid``` function, which is expected to strengthen object information in the image.   

In order to extract the background class and the class to be detected with different weight sets, a ```1x1 conv``` layer was added to obtain each feature map.   

Additionally, the output value was adjusted with the activation function ```f''``` to derive an output value close to both extremes. The functions used at this time were compared and experimented with ```none```, ```tanh```, ```ReLU```, ```LeakyReLU```, ```SeLU```, and ```GeLU```, which can output up to negative numbers.    

--------------------------------------------

## Dataset   
+ All images were preprocessed with 224 by 224 resolution.    

|Dataset|Train set|Valid set|Test set|
|:---:|:---:|:---:|:---:|
|[K-Vasir](https://paperswithcode.com/dataset/kvasir)|600|200|200|
|[Ottawa](https://ieeexplore.ieee.org/document/8506600)|558|181|185|      
  
## Evaluation   

$$Dice\ Score = 2 * \frac{Y \cap Y<sub>pred</sub>}{Y \cup Y<sub>pred</sub>}$$    

$$Dice\ CELoss = (1 - Dice\ Score) + Cross\ entropy$$   

---------------------

## **Optimal Algorithm Structure Comparison Experimental Method**     

A. Model of basic structure (top structure in Figure 2)    
B. Model of basic structure + b (only the output layer structure in Figure 2 changes)    
C. Model of basic structure + a + b (bottom structure in Figure 2)    
D. Modifying module a in the lower structure of Figure 2 (sig & tanh)     

### Figure 3   
<p align="center"><img src="https://user-images.githubusercontent.com/104747868/235351631-004a47bb-9cc5-4e05-b1f1-573143aa6a9b.png" width="600" height="500"><\   
 
### Fine tuning results by model   

<table>
<thead>
<tr>
<th style="text-align:center">Data Set</th>
<th colspan='5'>K-Vasir</th>
<th colspan='5'>Ottawa</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:center">Model</td>
<td style="text-align:center">FCN</td>
<td style="text-align:center">Unet</td>
<td style="text-align:center">DeepLabV3+</td>
<td style="text-align:center">FCBFormer</td>
<td style="text-align:center">Avg</td>
<td style="text-align:center">FCN</td>
<td style="text-align:center">Unet</td>
<td style="text-align:center">DeepLabV3+</td>
<td style="text-align:center">FCBFormer</td>
<td style="text-align:center">Avg</td>
</tr>
<tr>
<td style="text-align:center">A</td>
<td style="text-align:center">0.896</td>
<td style="text-align:center">0.843</td>
<td style="text-align:center">0.891</td>
<td style="text-align:center">0.912</td>
<td style="text-align:center">0.886</td>
<td style="text-align:center">0.919</td>
<td style="text-align:center">0.916</td>
<td style="text-align:center">0.922</td>
<td style="text-align:center">0.938</td>
<td style="text-align:center">0.924</td>
</tr>
<tr>
<td style="text-align:center">B</td>
<td style="text-align:center"><strong>0.900<br>None</strong></td>
<td style="text-align:center">0.858<br>ReLu</td>
<td style="text-align:center">0.896<br>SeLu</td>
<td style="text-align:center">0.911<br>SeLu</td>
<td style="text-align:center">0.891</td>
<td style="text-align:center"><strong>0.922<br>Leaky ReLu</strong></td>
<td style="text-align:center">0.919<br>None</td>
<td style="text-align:center"><strong>0.923<br>SeLu</strong></td>
<td style="text-align:center">0.942<br>Leaky ReLu</td>
<td style="text-align:center">0.927</td>
</tr>
<tr>
<td style="text-align:center">C</td>
<td style="text-align:center">0.899<br>Leaky ReLu</td>
<td style="text-align:center"><strong>0.872<br>SeLu, GeLu</strong></td>
<td style="text-align:center"><strong>0.898<br>ReLu</strong></td>
<td style="text-align:center"><strong>0.919<br>Leaky ReLu</strong></td>
<td style="text-align:center"><strong>0.897</strong></td>
<td style="text-align:center">0.921<br>ReLu</td>
<td style="text-align:center"><strong>0.926<br>SeLu</strong></td>
<td style="text-align:center"><strong>0.923<br>SeLu, GeLu</strong></td>
<td style="text-align:center"><strong>0.944<br>SeLu</strong></td>
<td style="text-align:center"><strong>0.929<br></strong></td>
</tr>
<tr>
<td style="text-align:center">D</td>
<td style="text-align:center">0.886<br>GeLu</td>
<td style="text-align:center">0.867<br>Leaky ReLu</td>
<td style="text-align:center">0.896<br>Leaky ReLu</td>
<td style="text-align:center">0.915<br>SeLu</td>
<td style="text-align:center">0.891</td>
<td style="text-align:center">0.921<br>ReLu</td>
<td style="text-align:center">0.915<br>SeLu</td>
<td style="text-align:center">0.917<br>ReLu</td>
<td style="text-align:center">0.941<br>GeLu</td>
<td style="text-align:center">0.924</td>
<td></td>
</tr>
</tbody>
</table>


## Conclusion   
+ As shown in the table above, when comparing the performance for the designed experiment, **the average performance of the version 'C Model'** was the best across the entire dataset, especially when the output layer function was in the **ReLu series**.   
+ **In all structures, output layer fine tuning showed performance improvement.**   
+ The addition of ```CBAM attention modules``` was also effective enough to **increase** average performance.   
+ The fine tuning method proposed in this paper is expected to have a greater range of performance improvement in datasets with a larger number than the dataset used in the experiment.
