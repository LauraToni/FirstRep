import os
from glob import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import pandas as pd
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout,Conv3D, Input, Dense, MaxPooling3D, BatchNormalization, ReLU
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, MaxPool3D, GlobalAveragePooling3D
from tensorflow.keras.models import Model, load_model
try:
    import nibabel as nib
except:
    raise ImportError('Install NIBABEL')
from data_augmentation import VolumeAugmentation

#pylint: disable=invalid-name
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

def read_dataset(dataset_path_AD, dataset_path_CTRL, x_id ="AD-", y_id="CTRL-"):
    """
    load images from NIFTI directory
    Parameters
    ----------
    dataset_path_AD: str
        directory path for AD images
    dataset_path_CTRL: str
        directory path for CTRL images
    x_id: str
        identification string in the filename of AD images
    y_id: str
        identification string in the filename of CTRL images

    Returns
    -------
    X : np.array
        array of AD and CTRL images data
    Y: np.array
        array of labels

    fnames_AD: list (?)
        list containig AD images file names
    fnames_CTRL: list (?)
        list containig CTRL images file names

    """
    fnames_AD = glob(os.path.join(dataset_path_AD, f"*{x_id}*.nii"  ))
    fnames_CTRL= glob(os.path.join(dataset_path_CTRL, f"*{y_id}*.nii"  ))
    X = []
    Y = []
    for fname_AD in fnames_AD:
        X.append(nib.load(fname_AD).get_fdata())
        Y.append(1)
    for fname_CTRL in fnames_CTRL:
        X.append(nib.load(fname_CTRL).get_fdata())
        Y.append(0)
    return np.array(X), np.array(Y), fnames_AD, fnames_CTRL

if __name__=='__main__':

dataset_path_AD_ROI = "AD_CTRL/AD_s3"
dataset_path_CTRL_ROI = "AD_CTRL/CTRL_s3"
dataset_path_metadata = "AD_CTRL_metadata_labels.csv"

"""
Import csv metadata
"""

df = pd.read_csv(dataset_path_metadata, sep=',')
head=df.head()
print(head)

#count the entries grouped by the diagnostic group
print(df.groupby('DXGROUP')['ID'].count())

features = ['DXGROUP', 'ID']
print(df[features])


X_o, Y, fnames_AD, fnames_CTRL = read_dataset(dataset_path_AD_ROI, dataset_path_CTRL_ROI)

#Da far diventare un Unittest
print(X_o.shape, Y.shape)
print(X_o.min(), X_o.max(), Y.min(), Y.max())

#Normalization of intensity voxel values
X_o=X_o/X_o.max()


#X=X_o[:,36:86,56:106,24:74] #ippocampo
#X=X_o[:,11:109,12:138,24:110] #bordi neri precisi
X=X_o[:,20:100,20:130,20:100]
#X=X_o
print(X.shape, Y.shape)

"""
Divide the dataset in train, validation and test in a static way

"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

#Da fare diventare un Unittest
print(f'X train shape: {X_train.shape}, X test shape: {X_test.shape}')
print(f'Y train shape: {Y_train.shape}, Y test shape: {Y_test.shape}')

'''
Data augmentation
'''

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=24)
mass_gen = VolumeAugmentation(X_train, Y_train, shape=(X.shape[1], X.shape[2], X.shape[3]))
array_img, labels = mass_gen.augment()

def stack_train_augmentation(img, img_aug, lbs, lbs_aug):
    """
    Creates an array containing both original and augmented images. Does the same with their label
    Parameters
    ----------
    img : 4D np.array
        array containing the images used for the training

    img_aug: 4D np.array
        array containing the augmented images used for the training
    lbs: np.array
        array containing the original image labels
    lbs_aug: np.array
        array containing the augmented image labels
    Returns
    -------
    img_tot : np.array
        array cointaing both original and augmented images
    lbs_tot : np.array
        array containing  original and augmented image labels

    """
    img_tot=np.append(img, img_aug, axis=0)
    lbs_tot=np.append(lbs, lbs_aug, axis=0)
    return img_tot, lbs_tot

X_train_tot, Y_train_tot=stack_train_augmentation(X_train, array_img, Y_train, labels)

def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = Conv3D(filters=8, kernel_size=3, activation="relu", kernel_regularizer='l2')(inputs)
    x = MaxPool3D(pool_size=2,  strides=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=16, kernel_size=3, activation="relu", kernel_regularizer='l2')(x)
    x = MaxPool3D(pool_size=2, strides=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=32, kernel_size=3, activation="relu", kernel_regularizer='l2')(x)
    x = MaxPool3D(pool_size=2,  strides=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=64, kernel_size=3, activation="relu", kernel_regularizer='l2')(x)
    x = MaxPool3D(pool_size=2,  strides=2)(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(units=64, activation="relu", kernel_regularizer='l2')(x)
    #x=Flatten()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


def get_model_art(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = Conv3D(filters=8, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(inputs)
    x= ReLU()(x)
    x = Conv3D(filters=8, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(x)
    x = BatchNormalization()(x)
    x= ReLU()(x)
    x = MaxPool3D(pool_size=2,  strides=2)(x)

    x = Conv3D(filters=16, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(x)
    x= ReLU()(x)
    x = Conv3D(filters=16, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(x)
    x = BatchNormalization()(x)
    x= ReLU()(x)
    x = MaxPool3D(pool_size=2,  strides=2)(x)

    x = Conv3D(filters=32, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(x)
    x= ReLU()(x)
    x = Conv3D(filters=32, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(x)
    x = BatchNormalization()(x)
    x= ReLU()(x)
    x = MaxPool3D(pool_size=2,  strides=2)(x)

    x = Conv3D(filters=64, kernel_size=3, activation="relu", kernel_regularizer='l1_l2')(x)
    x= ReLU()(x)
    x = Conv3D(filters=64, kernel_size=3, activation="relu", kernel_regularizer='l2')(x)
    x = BatchNormalization()(x)
    x= ReLU()(x)
    x = MaxPool3D(pool_size=2,  strides=2)(x)
    '''
    x = Conv3D(filters=128, kernel_size=3, activation="relu", kernel_regularizer='l2')(x)
    x= ReLU()(x)
    x = Conv3D(filters=128, kernel_size=3, activation="relu", kernel_regularizer='l2')(x)
    x = BatchNormalization()(x)
    x= ReLU()(x)
    x = MaxPool3D(pool_size=2,  strides=1)(x)
    '''
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model_art(width=X.shape[1], height=X.shape[2], depth=X.shape[3])
model.summary()

#initial_learning_rate = 0.001
ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate), loss='binary_crossentropy', metrics=['MAE'])

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "model.{epoch:02d}-{val_MAE:.4f}_C8_C8_C16_C16_C32_C32_D32_Hipp_art.h5", save_best_only=True
)
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1)

#And now let's fit it to our data.
#The sample is automatically split in two so that 50% of it is used for validation and the other half for training

history=model.fit(X_train_tot,Y_train_tot, validation_split=0.1, batch_size=32, shuffle=TRUE, epochs=20, callbacks=[early_stopping, ReduceLROnPlateau])

#history contains information about the training.
#We can now now show the loss vs epoch for both validation and training samples.

print(history.history.keys())
plt.plot(history.history["val_loss"])
plt.plot(history.history["loss"])
plt.plot(history.history["MAE"])
plt.plot(history.history["val_MAE"])
plt.legend()
plt.yscale('log')
plt.show()

#from keras.callbacks import ModelCheckpoint

history = model.fit(X_train_tot,Y_train_tot, validation_split=0.1, batch_size=32, epochs=10, callbacks=[checkpoint_cb, ReduceLROnPlateau])

#Definiamo la funzione per calcolare l'indice di DICE
def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

idx=67
xtrain = X_train_tot[idx][np.newaxis,...]
ytrain = Y_train_tot[idx][np.newaxis,...]
print(Y_train_tot[idx].shape, ytrain.shape)

ypred = model.predict(xtrain).squeeze()>0.1
ytrue = Y_train_tot[idx].squeeze()

dice_value = dice(ypred, ytrue)
print(f'Indice di DICE:{dice_value}')

print(ypred.shape, ytrue.shape)

'''
def dice_vectorized(pred, true, k = 1):
    intersection = 2.0 *np.sum(pred * (true==k), axis=(1,2,3))
    dice = intersection / (pred.sum(axis=(1,2,3)) + true.sum(axis=(1,2,3)))
    return dice
dice_vec=dice_vectorized(model.predict(X_train)>0.1, Y_train)
dice_vec_train_mean=dice_vectorized(model.predict(X_train)>0.1, Y_train).mean()
dice_vec_test_mean=dice_vectorized(model.predict(X_test)>0.1, Y_test).mean()
print(f'Indice di DICE vettorizzato medio train e medio test:{dice_vec_train_mean};{dice_vec_test_mean}')

'''
#use model to predict probability that given y value is 1
y_score = model.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_score)
#calculate AUC of model
auc = metrics.roc_auc_score(Y_test, y_score)
#print AUC score
print(auc)
#plot roc_curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % auc,)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()
