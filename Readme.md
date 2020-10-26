# Introduction  
At the beginning, I wanted to generate face images which I am able to change the facial attributes as a "Capstone Design 1" subject's goal. 
    But I didn't have enough knowledge about generative model to achieve the goal.
    That's why I started this project. I made this 'Simpson Generator' using DC-GAN, in order to understand how GAN works and to check whether I understand that or not. 

# How to use my code  
- In order to use my code, just follow this :  
```
git clone -b master https://github.com/Natural-Goldfish/SimpsonGenerator.git  
cd SimpsonGenerator\\SimpsonGenerator
python main.py --mode {}
```
**❗You must choose wihch mode you will run between 'train' or 'test'.**</br></br>

- For more specific information how to run my code, you could run :
```
python main.py -h
```

# Requirements
```
- python 3.7.1
- pytorch 1.6.0
- opencv 4.4.0
- numpy 1.15.4
```

# Project Structure
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

# Dataset  
I used "Simpsons Faces" dataset in Kaggle. If you search it, you can download it easily.

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

In this project, the cropped images are only used and those images are put into ``` 'data\images' ```
These are cropped images, so you don't need to additional working to make a dataset for training.
    
# Settings  
- **Model Structure**  
I followed DC-GAN paper's model architecture</br></br>
- **Loss**  
I used ```BCELoss``` in pytorch</br></br>
- **Optimizer**  
I used ```Adam optimizer``` with the β1 and β2 of default values. ```Learning rate 2e-4```

# Train  
- **Data Argumentation**  
I performed data argumentation to make model more stable and to complement the small dataset. Techniques applied here are resize, normalization, horizontal flip with random probability
I trained the model for 400 epochs about the dataset by 64 batch size. You can find this pre-trained model's parameter file in ```'data\models'```  </br></br>
- If you want to train this model from beginning, you could run :  
```python main.py --mode train```  </br></br>
- If you want to train pre-trained model, you could run :  
```python main.py --mode train --model_load_flag --generator_load_name {} --discriminator_load_name {}```

# Test  
You can generate images using pre-trained model, which are saved in 'data\generated_images'!  
- Just run :  
```python main.py --mode test```  </br></br>
- Also you can choice the number of images to generate by changing '--generate_numbers', you could run :  
```python main.py --mode test --generate_numbers {}```  </br></br>
- If you want to change the directory as well, you could run :  
```python main.py --mode test --generating_model_name {} --image_save_path {} --generate_numbers {}```

# Results  
Some generated images are shown below :

