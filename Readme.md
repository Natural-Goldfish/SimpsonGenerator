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
I used _"Simpsons Faces"_ dataset in Kaggle. If you search it, you can download it easily.

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
</br></br>

# Settings  
- **Model Structure**</br></br>
I followed DC-GAN paper's model architecture</br></br>
- **Loss**</br></br>
I used _```BCELoss```_ in pytorch</br></br>
- **Optimizer**</br></br>
I used _```Adam optimizer```_ with the β1 and β2 of default values. _```Learning rate : 2e-4```_
</br></br>

# Train  
- **Data Argumentation**</br></br>
I performed data argumentation to make model more stable and to complement the small dataset. Techniques applied here are _resize_, _normalization_, _horizontal flip with random probability_.
I trained the model for _400 epochs_ about the dataset by _64 batch size_. You can find this pre-trained model's parameter file in ```'data\models'```</br></br>
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
You can generate images using pre-trained model, which are saved in ```'data\generated_images'```  
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
![](https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img0.jpg?raw=true)
![](https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img1.jpg?raw=true)
![](https://github.com/Natural-Goldfish/SimpsonGenerator/blob/master/SimpsonGenerator/data/generated_images/Generated_img2.jpg?raw=true)


