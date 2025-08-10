# Image Segmentation on Cityscapes Dataset

## Objective

The motivation for this project is to test the possibility of CNN image segmentation using an ordinary laptop with 8GB visual memory and explore methods to increase model accuracy and efficiency within this limitation. 

## baseline model
I choose the standard UNet and the baseline model using the setup from [kerrgarr](https://github.com/kerrgarr/SemanticSegmentationCityscapes)

## dataset
**Cityscapes: Semantic Understanding of Urban Street Scenes.** https://www.cityscapes-dataset.com/downloads

The dataset contains fine annotated daytime cityscape dataset in Germany with various weather conditions.

Annotated features were classified into 8 groups (human, viechle, construction etc.)

<img width="547" height="280" alt="image" src="https://github.com/user-attachments/assets/99638975-8024-42e4-aaaa-f185b1aedc26" />


## Evaluation Criteria 
We choose Intersaction Over Union (IoU) as the measurement of accuracy. Based on the state of art performance, we choose IoU > 0.5 as the acceptance threshold. That is, if IoU > 0.5, we consider the model accurately identifies the object.

<img width="589" height="417" alt="image" src="https://github.com/user-attachments/assets/ad132bf3-0f7f-4fe2-8529-0d16f3f4f3f4" />

___________________________________
## initial results
The baseline model shows good performance on background detection (e.g. sky, road, trees)

While the detection for smaller objects such as viechle and human are consistently challenging

<img width="470" height="345" alt="image" src="https://github.com/user-attachments/assets/f4fa5e4a-9997-4953-a6de-b62b86463863" />

## Modifications

### resolution adjustment:
Simply increase resolution from 128 * 128 to 256 * 256 would result in 25% increase in global accuracy (0.4 to 0.5)

Yet due to computation limitation 256 resolution is the highest our laptop could handle

### weight adjustment:
We tried to gradually increase the focal loss weight for viechle and person class in order to increase model's detection priority.

Result shows that increasing weight would lead to better viechle/person detection with the cost of slight decrease in accuracy of other classes. However, the model deteriorates when the weight is set too high.

<img width="1520" height="826" alt="image" src="https://github.com/user-attachments/assets/4705b230-47ac-45b0-8529-becab5a07651" />

### Data Augmentation
We employed color jitter, gray scale, horizontal flip and random corp in the hope to increase model's robustness in dealing with different color and textures.

Though no significant improvements were shown in ordinary cases, we do find data augmentation useful for peculiar cases. For example, model with augmentation showed better performance in inverted and black-and-white images compared to the baseline model. 

* Please feel free to check on my presentation slides for more info and results!

[Presentation Slides](https://github.com/Pafuuuu/SemanticSegmentationCityscapes/blob/main/Psych186B_Presentation%20.pdf)








