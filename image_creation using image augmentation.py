#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Create an ImageDataGenerator object with augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Path to the dataset directory containing multiple images
dataset_path = r'C:\Users\soura\OneDrive\Desktop\skin-disease-datasaet\train_set\VI-chickenpox'#generate image loaction

# Directory to save augmented images
augmented_dir = r'C:\Users\soura\OneDrive\Desktop\skin-disease-datasaet\augmented_images'#saving image location
if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

# Iterate over each image in the dataset directory
for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(dataset_path, filename)
        img = load_img(img_path)  # Load the image
        x = img_to_array(img)  # Convert the image to a Numpy array
        x = x.reshape((1,) + x.shape)  # Reshape to (1, height, width, channels)

        # Generate and save augmented images
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=augmented_dir, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i >= 20:  # Generate 20 augmented images per original image
                break

# Check if there are any augmented images
augmented_images = os.listdir(augmented_dir)
if len(augmented_images) == 0:
    print("No augmented images found. Please check the augmentation process.")
else:
    # Display some augmented images
    num_images_to_display = min(5, len(augmented_images))
    for i in range(num_images_to_display):
        img = mpimg.imread(os.path.join(augmented_dir, augmented_images[i]))
        plt.imshow(img)
        plt.axis('off')  # Hide axis
        plt.show()


# In[ ]:




