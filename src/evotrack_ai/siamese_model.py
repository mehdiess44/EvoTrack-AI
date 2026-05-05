"""Keras Siamese model architecture for MRI evolution classification."""

import tensorflow as tf


INPUT_SHAPE = (224, 224, 3)


def build_backbone(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    trainable: bool = False,
) -> tf.keras.Model:
    """Build a MobileNetV2 backbone with an automatic ImageNet fallback.

    Args:
        input_shape: Input image shape.
        trainable: Whether the backbone weights should be trainable.

    Returns:
        A Keras MobileNetV2 model.
    """
    try:
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    except Exception as error:
        print(
            "Could not load ImageNet weights for MobileNetV2. "
            "Falling back to randomly initialized weights."
        )
        print(f"Reason: {error}")
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_shape=input_shape,
        )

    backbone.trainable = trainable

    return backbone


def absolute_difference(tensors):
    """Return the absolute difference between two tensors.

    Args:
        tensors: A list or tuple containing exactly two tensors.

    Returns:
        ``tf.abs(tensor_a - tensor_b)``.

    Raises:
        ValueError: If exactly two tensors are not provided.
    """
    if len(tensors) != 2:
        raise ValueError("absolute_difference expects exactly two tensors.")

    tensor_a, tensor_b = tensors

    return tf.abs(tensor_a - tensor_b)


def build_siamese_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    backbone_trainable: bool = False,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """Build and compile a Siamese binary classifier with shared weights.

    Args:
        input_shape: Shape of each input image.
        backbone_trainable: Whether the shared backbone should be trainable.
        learning_rate: Adam optimizer learning rate.

    Returns:
        A compiled Keras Siamese model.
    """
    input_t0 = tf.keras.Input(shape=input_shape, name="image_t0")
    input_t1 = tf.keras.Input(shape=input_shape, name="image_t1")

    backbone = build_backbone(
        input_shape=input_shape,
        trainable=backbone_trainable,
    )

    features_t0 = backbone(input_t0)
    features_t1 = backbone(input_t1)

    difference = tf.keras.layers.Lambda(
        absolute_difference,
        name="absolute_difference",
    )([features_t0, features_t1])

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(difference)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense_128")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = tf.keras.Model(
        inputs=[input_t0, input_t1],
        outputs=output,
        name="evotrack_siamese_model",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def test_forward_pass(
    model: tf.keras.Model,
    batch_size: int = 2,
) -> tf.Tensor:
    """Run a simple forward pass with zero-valued image tensors."""
    image_t0 = tf.zeros((batch_size, *INPUT_SHAPE), dtype=tf.float32)
    image_t1 = tf.zeros((batch_size, *INPUT_SHAPE), dtype=tf.float32)

    predictions = model([image_t0, image_t1], training=False)

    return predictions


if __name__ == "__main__":
    siamese_model = build_siamese_model()
    siamese_model.summary()

    predictions = test_forward_pass(siamese_model)

    print(f"predictions shape: {predictions.shape}")
    print(f"predictions values: {predictions.numpy()}")

    assert predictions.shape == (2, 1)
