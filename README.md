# Introduction  
At the beginning, I wanted to generate face images which I am able to change the facial attributes as a "Capstone Design 1" subject's goal. 
    But I didn't have enough knowledge about generative model to achieve the goal.
    That's why I started this project. I made this 'Simpson Generator' using DC-GAN, in order to understand how GAN works and to check whether I understand that or not.  
</br></br>
# How to use my code  
- In order to use my code, just follow this :  
```
    git clone -b master https://github.com/Natural-Goldfish/SimpsonGenerator.git  
```
```
    cd SimpsonGenerator\\SimpsonGenerator
```  
**❗ You must choose wihch mode you will run between _'train' or 'test'_.**
```
    python main.py --mode {}
```
- For more specific information how to run my code, you could run :
```  
    python main.py -h  
```
</br></br>

# Requirements
```
    - python 3.7.1
    - pytorch 1.6.0
    - opencv 4.4.0
    - numpy 1.15.4  
```
</br></br>

# Project Structure  
This show how my project files stored. The structure looks like below :  
```
    SimpsonGenerator
    ├──data
    │   ├──generated_images
    │   │   ├──Generated_img0.jpg
    │   │   ├──Generated_img1.jpg
    │   │   └──...
    │   ├──images
    │   │   ├──1.png
    │   │   ├──2.png
    │   │   └──...
    │   └──models
    │       ├──generator_{epoch}_checkpoint.pth
    │       └──discriminator_{epoch}_checkpoint.pth
    ├──src
    │   ├──__init__.py
    │   ├──data_argumentation.py
    │   ├──dataset.py
    │   ├──network.py
    │   └──utils.py
    ├──__init__.py
    ├──main.py
    ├──training.py
    └──test.py  
```
</br></br>

# Dataset  
I used  _"[Simpsons Faces](https://www.kaggle.com/kostastokis/simpsons-faces)"_ dataset in Kaggle. The dataset structure looks like below :  

```
    Dataset
    ├──cropped
    │   ├──1.png
    │   ├──2.png
    │   └──...
    └──simplified
        ├──1.png
        ├──2.png
        └──...  
```  

In this project, only the cropped images are used and are put into _``` 'data\images' ```_
These are cropped images, so you don't need to additional working to make a dataset for training.</br></br>
<p align="center"><img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/images/1.png' width = "200px" height = "200px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/images/4.png' width = "200px" height = "200px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/images/3.png' width = "200px" height = "200px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/images/14.png' width = "200px" height = "200px"/></p>
</br></br>

# Settings  
- **Model Structure**</br></br>
I followed DC-GAN paper's model architecture</br></br>
- **Loss**</br></br>
I used _BCELoss_ in pytorch</br></br>
- **Optimizer**</br></br>
I used _Adam optimizer_ with the β1 and β2 of default values. _Learning rate : 2e-4_</br></br>
- **Data Argumentation**</br></br>
I performed data argumentation to make model more stable and to complement the small dataset. Techniques applied here are _resize_, _normalization_, _horizontal flip with random probability_.  
</br></br>

# Train  
I trained the model for _400 epochs_ about the dataset by _64 batch size_. You can find this pre-trained model's parameter file in _```'data\models'```_  
The update cycle of disciminator model for each batch size is a little bit changed comapred to the paper.  
| Epoch | Discriminator | Generator |  
|---|---|---|
| 0 ~ 50 | 2 | 1 |
| 50 ~ end | 1 | 1|  
- If you want to train this model from beginning, you could run :  
```
python main.py --mode train
```
- If you want to train pre-trained model, you could run :  
``` 
python main.py --mode train --model_load_flag --generator_load_name {} --discriminator_load_name {}
```
</br></br>

# Test  
You can generate images using pre-trained model, which are saved in _```'data\generated_images'```_  
- If you want to see a generated image which pre-trained model make, just run :  
``` 
python main.py --mode test
```  
- Also you can choice the number of images to generate by changing _**'--generate_numbers'**_, you could run :  
``` 
python main.py --mode test --generate_numbers {}
```  
- If you want to change the directory as well, you could run :  
``` 
python main.py --mode test --generating_model_name {} --image_save_path {} --generate_numbers {}
```  
</br></br>

# Results  
Some generated images are shown below :</br></br>
<p align="center"><img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img0.jpg' width = "100px" height = "100px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img2.jpg' width = "100px" height = "100px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img3.jpg' width = "100px" height = "100px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img4.jpg' width = "100px" height = "100px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img5.jpg' width = "100px" height = "100px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img6.jpg' width = "100px" height = "100px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img7.jpg' width = "100px" height = "100px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img8.jpg' width = "100px" height = "100px"/>
<img src='https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img9.jpg" width = "100px" height = "100px"/></p>


