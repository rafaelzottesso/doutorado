# Configurações de locais
base_spec = './base_audio_spec/'
base_destino = './base_dinâmica/'
############################################################
# Usar o ImageDataGenerator para dividir um diretório em treino e teste
############################################################
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen = ImageDataGenerator(validation_split=0.2)

train = data_gen.flow_from_directory(
        base_spec,
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
        subset="training",
    )

test = data_gen.flow_from_directory(
        base_spec,
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
        subset="validation",
    )
############################################################

############################################################
# Copiar os arquivos obtidos para um novo diretório
############################################################
from shutil import copyfile
import os

print('Copiando base de treino.')
for img in train.filepaths:
    dest = base_destino+"treino/{}".format(img.replace(base_spec, ""))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    copyfile(img, dest)

print('Copiando base de teste.')
for img in test.filepaths:
    dest = base_destino+"teste/{}".format(img.replace(base_spec, ""))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    copyfile(img, dest)
############################################################