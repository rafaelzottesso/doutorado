# https://www.pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/


# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np


treino_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,)

train = treino_gen.flow_from_directory(
        './treino/',
        target_size=(400, 348),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir="",
        save_prefix="",
        save_format="png",
        follow_links=False,
    )

teste_gen = ImageDataGenerator()

test = teste_gen.flow_from_directory(
        './teste/',
        target_size=(400, 348),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir="",
        save_prefix="",
        save_format="png",
        follow_links=False,
    )


baseModel = ResNet50(weights="./resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False, input_tensor=Input(shape=(400, 348, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(47, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False
    
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


H = model.fit_generator(
	train,
	steps_per_epoch=12,
	validation_data=test,
	validation_steps=3,
	epochs=100)