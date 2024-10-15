# -*- coding: utf-8 -*-
"""
Project Goal: Driver behavior monitoring image classification
File Goal: Data cleaning and preprocessing
"""

# Import necessary libraries
import os
import tensorflow as tf
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Parsing function for the dataset
def _parse_function(proto):
    keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string),
        'image/object/class/text': tf.io.VarLenFeature(tf.string)
    }
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    image = tf.io.decode_image(parsed_features['image/encoded'])
    labels = parsed_features['image/object/class/text'].values
    return image, labels

# Dataset paths
base_path = 'DATA/3MDAD_TFRECORDS/'
train_path = os.path.join(base_path, 'train/safe-driving.tfrecord')
valid_path = os.path.join(base_path, 'valid/safe-driving.tfrecord')
test_path = os.path.join(base_path, 'test/safe-driving.tfrecord')

# Load datasets
train_dataset = tf.data.TFRecordDataset(train_path)
valid_dataset = tf.data.TFRecordDataset(valid_path)
test_dataset = tf.data.TFRecordDataset(test_path)

# Class names for driving behaviors
class_names = [
    "Adjusting radio", "Doing hair and makeup", "Drinking using left hand", 
    "Drinking using right hand", "Fatigue and somnolence", "GPS operating", 
    "Having picture", "Reaching behind", "Singing or dancing", "Smoking", 
    "Talking phone using left hand", "Talking phone using right hand", 
    "Talking to passenger", "Writing message using left hand", 
    "Writing message using right hand", "safe-driving"
]

# Find and display one image per class
def display_class_samples(dataset, class_names, num_classes=16):
    found_images = {}
    for img, labels in dataset:
        for label in labels.numpy():
            decoded_label = label.decode('utf-8')
            if decoded_label not in found_images:
                found_images[decoded_label] = img.numpy()
            if len(found_images) == num_classes:
                break
        if len(found_images) == num_classes:
            break

    # Display images in a grid layout
    grid_size = 4
    grid_image = np.zeros((grid_size * 224, grid_size * 224, 3), dtype=np.uint8)
    for i, (class_name, img) in enumerate(found_images.items()):
        x = i % grid_size
        y = i // grid_size
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (224, 224))
        img = cv2.putText(img, class_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        grid_image[y * 224:(y + 1) * 224, x * 224:(x + 1) * 224, :] = img

    cv2_imshow(grid_image)

# Parse and display a sample of the training dataset
train_dataset_parsed = train_dataset.map(_parse_function)
display_class_samples(train_dataset_parsed, class_names)

# Count and validate the number of images
def count_images(dataset, expected_count, name):
    actual_count = sum(1 for _ in dataset)
    print(f'[{name}] Expected images: {expected_count}, Actual images: {actual_count}')

count_images(train_dataset, 4803, "Training")
count_images(valid_dataset, 528, "Validation")
count_images(test_dataset, 2357, "Test")

# Function to check for missing metadata
def check_missing_metadata(dataset):
    missing_metadata_count = 0
    for record in dataset:
        _, labels = _parse_function(record)
        if not labels.numpy().size:
            missing_metadata_count += 1
    if missing_metadata_count:
        print(f'Missing metadata found in {missing_metadata_count} images.')
    else:
        print('No missing metadata found.')

check_missing_metadata(train_dataset)

# Model preparation using VGG16
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint

# Convert labels to integers
def labels_to_integers(labels):
    labels = [label.decode('utf-8') for label in labels.numpy()]
    return [class_names.index(label) for label in labels]

# Parse function with label conversion
def _parse_function_with_labels_as_integers(proto):
    keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
    }
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    img = tf.image.decode_jpeg(parsed_features['image/encoded'], channels=3)
    labels = parsed_features['image/object/class/text'].values
    labels_int = tf.py_function(labels_to_integers, [labels], tf.int32)
    return img, labels_int

# Preprocess input data
def preprocess_input_data(dataset, parse_func, batch_size):
    dataset = dataset.map(parse_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x, y: (tf.image.resize(x, (224, 224)), y))
    dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

batch_size = 32
train_dataset_preprocessed = preprocess_input_data(train_dataset, _parse_function_with_labels_as_integers, batch_size)
valid_dataset_preprocessed = preprocess_input_data(valid_dataset, _parse_function_with_labels_as_integers, batch_size)
test_dataset_preprocessed = preprocess_input_data(test_dataset, _parse_function_with_labels_as_integers, batch_size)
