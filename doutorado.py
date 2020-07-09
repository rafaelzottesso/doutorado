import tensorflow as tf

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,)

train = data_gen.flow_from_directory(
        './base_juliano_segmentada_automatico/',
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
        subset="training",
    )

print(train[1][1].shape)


test = data_gen.flow_from_directory(
        './base_juliano_segmentada_automatico/',
        target_size=(400, 348),
        color_mode="rgb",
        classes=None,
        class_mode="categorical",
        batch_size=32,
        shuffle=True,
        seed=None,
        save_to_dir="./base_test/",
        save_prefix="",
        save_format="png",
        follow_links=False,
        subset="validation",
    )

# image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     )

# train = image_generator.flow_from_directory (
#         './base_juliano_segmentada_automatico/',
#         target_size=(400, 348),
#         color_mode="rgb",
#         classes=None,
#         class_mode="categorical",
#         batch_size=32,
#         shuffle=True,
#         seed=None,
#         save_to_dir=None,
#         save_prefix="",
#         save_format="png",
#         follow_links=False,
#         subset="training",
#     )

# test = image_generator.flow_from_directory (
#         './base_juliano_segmentada_automatico/',
#         target_size=(400, 348),
#         color_mode="rgb",
#         classes=None,
#         class_mode="categorical",
#         batch_size=32,
#         shuffle=True,
#         seed=None,
#         save_to_dir=None,
#         save_prefix="",
#         save_format="png",
#         follow_links=False,
#         subset="validation",
#     )


