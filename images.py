import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.densenet import preprocess_input, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import tensorflow.keras.backend as K
from PIL import Image
import numpy as np
import os

class FeatureExtractor():
    def __init__(self,
                 shape=[256, 256, 3]):
        self.shape = shape
        input_tensor = Input(shape)
        densenet = DenseNet121(input_tensor=input_tensor,
                               weights='imagenet',
                               include_top=False)
        out = densenet.output
        out = GlobalAveragePooling2D()(out)
        out = Lambda(lambda x: K.expand_dims(x, axis=-1))(out)
        out = AveragePooling1D(4)(out)
        out = Lambda(lambda x: x[:,:,0])(out)

        self.model = Model(input_tensor, out)

    def convert_to_square(self, img):
        return img.resize(self.shape[:2]) 

    def load_image_by_path(self, filepath):
        img = Image.open(filepath)
        img = self.convert_to_square(img)
        img = np.array(img).astype(np.float32)
        img = preprocess_input(img)
        if len(img.shape) == 2:
            img = np.repeat(np.expand_dims(img, axis=2), repeats=3, axis=2)
        return img

    def extract(self, filepath):
        img = self.load_image_by_path(filepath)
        return self.model.predict(np.expand_dims(img, axis=0))
