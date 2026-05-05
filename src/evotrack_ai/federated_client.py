"""Federated client simulation for one hospital node."""

import numpy as np
import tensorflow as tf


try:
    from evotrack_ai.federated_data import create_federated_datasets
    from evotrack_ai.siamese_model import build_siamese_model
except ModuleNotFoundError:
    from federated_data import create_federated_datasets
    from siamese_model import build_siamese_model


class FederatedClient:
    """Simulate a hospital client with local data and a local model."""

    def __init__(
        self,
        client_id: str,
        dataset: tf.data.Dataset,
        learning_rate: float = 1e-4,
    ) -> None:
        """Initialize one federated client.

        Args:
            client_id: Unique hospital client identifier.
            dataset: Local ``tf.data.Dataset``.
            learning_rate: Learning rate used to compile the local model.

        Raises:
            ValueError: If ``client_id`` or ``dataset`` is invalid.
        """
        if not isinstance(client_id, str) or not client_id.strip():
            raise ValueError("client_id must be a non-empty string.")

        if dataset is None:
            raise ValueError("dataset must not be None.")

        self.client_id = client_id
        self.dataset = dataset.map(
            _format_batch_for_keras,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        self.model = build_siamese_model(learning_rate=learning_rate)

    def set_global_weights(self, global_weights: list[np.ndarray]) -> None:
        """Replace local model weights with global server weights."""
        if not isinstance(global_weights, list) or not global_weights:
            raise ValueError("global_weights must be a non-empty list.")

        self.model.set_weights(global_weights)

    def train_local_epoch(
        self,
        steps_per_epoch: int = 5,
        epochs: int = 1,
        verbose: int = 0,
    ):
        """Train the local model for a small number of local steps."""
        history = self.model.fit(
            self.dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=verbose,
        )

        return history

    def get_local_weights(self) -> list[np.ndarray]:
        """Return the current local model weights."""
        return self.model.get_weights()

    def evaluate_local(
        self,
        steps: int = 3,
        verbose: int = 0,
    ) -> dict:
        """Evaluate the local model on the local dataset."""
        return self.model.evaluate(
            self.dataset,
            steps=steps,
            verbose=verbose,
            return_dict=True,
        )


def _format_batch_for_keras(
    image_1: tf.Tensor,
    image_2: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
    """Format a batch for a two-input Keras model."""
    return (image_1, image_2), label


def weights_have_changed(
    before_weights: list[np.ndarray],
    after_weights: list[np.ndarray],
) -> bool:
    """Return True if at least one weight array changed."""
    if len(before_weights) != len(after_weights):
        raise ValueError("Weight lists must have the same length.")

    for before_array, after_array in zip(before_weights, after_weights):
        if not np.array_equal(before_array, after_array):
            return True

    return False


if __name__ == "__main__":
    federated_datasets = create_federated_datasets(batch_size=8)
    dataset_a = federated_datasets["Client_A"]

    client = FederatedClient("Client_A", dataset_a)
    initial_weights = client.get_local_weights()
    client.train_local_epoch(steps_per_epoch=2, epochs=1, verbose=1)
    updated_weights = client.get_local_weights()
    weights_changed = weights_have_changed(initial_weights, updated_weights)

    print(f"client_id: {client.client_id}")
    print(f"number of weight arrays: {len(updated_weights)}")
    print(f"first weight shape: {updated_weights[0].shape}")
    print(f"weights changed: {weights_changed}")

    assert weights_changed is True
