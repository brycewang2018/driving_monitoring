## To replace the MobileNetV2 model with the CoCa (Contrastive Captioners) model, 
# we will need to adjust the structure to fit how CoCa processes images and 
# integrates vision and text understanding. Since you asked for a fine-tuned 
# CoCa model, we'll use the base of CoCa from Hugging Face's transformers library, 
# and make the necessary modifications. Note that CoCa has a different architecture
#  from MobileNet, and it integrates both image and text modalities.

# -*- coding: utf-8 -*-
"""
Goal: Driver behavior monitoring image classification
Model: CoCa (finetuned)
"""

import os
import pandas as pd
import numpy as np
from transformers import VisionEncoderDecoderModel, ViTImageProcessor
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Define the image and annotation paths
train_image_path = "DATA/3MDAD_CSV/train"
train_annot_path = "DATA/3MDAD_CSV/train/_annot_new.csv"
valid_image_path = "DATA/3MDAD_CSV/valid"
valid_annot_path = "DATA/3MDAD_CSV/valid/_annot_new.csv"
test_image_path = "DATA/3MDAD_CSV/test"
test_annot_path = "DATA/3MDAD_CSV/test/_annot_new.csv"

# Define the number of classes
num_classes = 11

# Load the annotations
train_df = pd.read_csv(train_annot_path)
valid_df = pd.read_csv(valid_annot_path)
test_df = pd.read_csv(test_annot_path)

# Define the image size (CoCa uses ViT architecture for image encoding)
img_size = (224, 224)

# Define the batch size
batch_size = 32

# Define the image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=train_image_path,
    x_col="filename",
    y_col="class",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

val_generator = valid_datagen.flow_from_dataframe(
    valid_df,
    directory=valid_image_path,
    x_col="filename",
    y_col="class",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=test_image_path,
    x_col="filename",
    y_col="class",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# Load the CoCa model (using ViT base for the image encoding part)
model_name = "microsoft/coca-base"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)

# Freeze the vision encoder layers, as they are pre-trained
for param in model.encoder.parameters():
    param.requires_grad = False

# Define an Adam optimizer
optimizer = Adam(lr=0.0001)

# Compile the model (with categorical crossentropy for multi-class classification)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

## Model Fitting
epochs = 100
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=val_generator)

# Saving the history
import pickle
pickle.dump(history.history, open("history/3MDAD_CSV", "wb"))

df_history = pd.DataFrame(history.history)
df_history.head()

df_history.to_csv('history/3MDAD_CSV/CoCa_history.csv')

# Plot the training and validation accuracy and loss curves
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('Model Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Validation'], loc='upper left')

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Create confusion matrix
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

confusion_matrix = tf.math.confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import accuracy_score, f1_score

accuracy = accuracy_score(y_true, y_pred_classes)
print("Accuracy:", accuracy)

f1 = f1_score(y_true, y_pred_classes, average='weighted')
print("F1 Score:", f1)
