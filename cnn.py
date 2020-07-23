# https://www.pyimagesearch.com/2020/04/27/fine-tuning-resnet-with-keras-tensorflow-and-deep-learning/


# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50, ResNet152V2, InceptionV3, DenseNet201
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# Variáveis de locais
dir_treino = './base_dinâmica/treino/'
dir_teste = './base_dinâmica/teste/'

treino_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,)

train = treino_gen.flow_from_directory(
        dir_treino,
        target_size=(224, 224),
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
        dir_teste,
        target_size=(224, 224),
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

# baseModel = ResNet50(
#     weights="./resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", 
#     include_top=False,
#     input_tensor=Input(shape=(224, 224, 3))
#     )


# baseModel = ResNet152V2(
#     include_top=False, 
#     weights="imagenet",
#     input_tensor=Input(shape=(224, 224, 3)),
#     )

# baseModel = InceptionV3(
#     include_top=False, 
#     weights="imagenet",
#     input_tensor=Input(shape=(224, 224, 3)),
#     )

baseModel2 = ResNet50(
    include_top=False, 
    weights="imagenet",
    input_tensor=Input(shape=(224, 224, 3)),
    )


# inputs = keras.Input(shape=(150, 150, 3)) x = base_model(inputs, training=False) x = keras.layers.GlobalAveragePooling2D()(x) outputs = keras.layers.Dense(1)(x)

# headModel = baseModel.output
headModel2 = baseModel2.output

# headModel = AveragePooling2D(pool_size=(7,7))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(256, activation="relu")(headModel)

# headModel = GlobalAveragePooling2D()(headModel)
# headModel = Dropout(0.2)(headModel)
# headModel = Dense(47, activation="softmax")(headModel)

headModel2 = GlobalAveragePooling2D()(headModel2)
headModel2 = Dropout(0.2)(headModel2)
headModel2 = Dense(47, activation="softmax")(headModel2)

# inputs = keras.Input(shape=(150, 150, 3))
# x = base_model(inputs, training=False)
# x = keras.layers.GlobalAveragePooling2D()(x)
# outputs = keras.layers.Dense(1)(x)

# model = Model(inputs=baseModel.input, outputs=headModel)
model2 = Model(inputs=baseModel2.input, outputs=headModel2)

# for layer in baseModel.layers:
# 	layer.trainable = False

for layer in baseModel2.layers:
	layer.trainable = False
    
# optimizer, SGD, outros
# especificar learning rate    
# early stop
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.compile(
#     loss="categorical_crossentropy", 
#     optimizer=Adam(learning_rate=0.001), #0.05 0.001 0.005
#     metrics=["accuracy"]
#     )


# H = model.fit_generator(
# 	train,
# 	steps_per_epoch=12,
# 	validation_data=test,
# 	validation_steps=3,
# 	epochs=100)

model2.compile(
    loss="categorical_crossentropy", 
    optimizer=Adam(learning_rate=0.001), #0.05 0.001 0.005
    metrics=["accuracy"]
    )


H2 = model2.fit_generator(
	train,
	steps_per_epoch=12,
	validation_data=test,
	validation_steps=3,
	epochs=100)