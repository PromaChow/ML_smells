import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from skimage.transform import resize

# Load the dataset
def load_data(path):
    X = np.load(os.path.join(path, 'train_images.npy'))
    y = np.load(os.path.join(path, 'train_masks.npy'))
    return X, y

X, y = load_data('path_to_dataset')

# Data Augmentation
def augment_data(X, y, augments=None):
    if augments is None:
        augments = [0, 1, 2]
    
    for i in range(len(X)):
        img = X[i].copy()
        mask = y[i].copy()

        for a in augments:
            if a == 0:  # Original
                yield (img, mask)
            elif a == 1:  # Horizontal Flip
                img = np.fliplr(img)
                mask = np.fliplr(mask)
                yield (img, mask)
            elif a == 2:  # Vertical Flip
                img = np.flipud(img)
                mask = np.flipud(mask)
                yield (img, mask)

# Train-Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# U-Net Model Implementation
def unet(input_size=(101, 101, 1)):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    bn2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    bn3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Decoder
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up4)
    bn4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(bn4)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up5)
    bn5 = BatchNormalization()(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(bn5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

model = unet()
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit_generator(
    augment_data(X_train, y_train),
    steps_per_epoch=len(X_train) // 4,
    validation_data=(X_val, y_val),
    epochs=20,
    verbose=1,
)

# Save and load model for evaluation
model.save('unet_salt.h5')
new_model = load_model('unet_salt.h5', custom_objects={'KerasLayer': tf.keras.layers.Conv2D})

# Predict on test data
def predict_masks(model, X_test):
    predictions = new_model.predict(X_test)
    return (predictions > 0.5).astype(int)

X_test = np.load(os.path.join('path_to_dataset', 'test_images.npy'))
y_pred = predict_masks(new_model, X_test)

# Post-processing and formatting
def rle_encode(mask_image):
    pixels = mask_image.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)

# Generate submission file
submission = pd.DataFrame()
for image_id, mask_image in zip(df_test.image_id.values, y_pred):
    rle_mask = rle_encode(mask_image)
    submission = submission.append({'id': image_id, 'rle_mask': rle_mask}, ignore_index=True)

submission.to_csv('submission.csv', index=False)
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the metadata
metadata = pd.read_csv('train_labels.csv')

# Split the data into training and validation sets
train_images, val_images = train_test_split(metadata, test_size=0.2, random_state=42)

def load_data(image_ids, path='train'):
    images = []
    labels = []
    for idx in image_ids['id']:
        img_path = os.path.join(path, idx)
        img = # Load the image using PIL or OpenCV
        label = image_ids.loc[image_ids['id'] == idx, 'label'].values[0]
        images.append(img)
        labels.append(label)
    return images, labels

# Load training data
train_images, train_labels = load_data(train_images)

# Load validation data
val_images, val_labels = load_data(val_images)

# Display the shapes of the loaded data
print(f"Training images shape: {len(train_images)}, Labels shape: {len(train_labels)}")
print(f"Validation images shape: {len(val_images)}, Labels shape: {len(val_labels)}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Save the model
model.save('cancer_detection_model.h5')