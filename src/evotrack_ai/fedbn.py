"""Federated Batch Normalization utilities for EvoTrack AI."""

import numpy as np
import tensorflow as tf


try:
    from evotrack_ai.federated_client import FederatedClient
    from evotrack_ai.federated_data import create_federated_datasets
    from evotrack_ai.federated_server import FederatedServer, validate_weight_lists
except ModuleNotFoundError:
    from federated_client import FederatedClient
    from federated_data import create_federated_datasets
    from federated_server import FederatedServer, validate_weight_lists


def get_batch_norm_weight_indices(model: tf.keras.Model) -> set[int]:
    """Return indices of BatchNormalization weights in ``model.get_weights()``.

    Nested Keras models are traversed recursively so BatchNormalization layers
    inside the MobileNetV2 backbone are included.
    """
    if model is None:
        raise ValueError("model must not be None.")

    batch_norm_indices: set[int] = set()
    current_index = 0

    def visit_layer(layer) -> None:
        nonlocal current_index

        if isinstance(layer, tf.keras.layers.BatchNormalization):
            num_weights = len(layer.get_weights())
            batch_norm_indices.update(
                range(current_index, current_index + num_weights)
            )
            current_index += num_weights
            return

        if isinstance(layer, tf.keras.Model):
            for sub_layer in layer.layers:
                visit_layer(sub_layer)
            return

        current_index += len(layer.get_weights())

    for model_layer in model.layers:
        visit_layer(model_layer)

    return batch_norm_indices


def get_non_batch_norm_weight_indices(model: tf.keras.Model) -> set[int]:
    """Return all model weight indices that do not belong to BatchNorm layers."""
    batch_norm_indices = get_batch_norm_weight_indices(model)
    all_indices = set(range(len(model.get_weights())))

    return all_indices - batch_norm_indices


def aggregate_fedbn_weights(
    client_weights: list[list[np.ndarray]],
    reference_model: tf.keras.Model,
    reference_global_weights: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Aggregate client weights with FedBN.

    BatchNormalization weights are not averaged. They are copied from the
    reference global model when available, otherwise from the first client.
    All non-BN weights are averaged with standard FedAvg.
    """
    validate_weight_lists(client_weights)

    batch_norm_indices = get_batch_norm_weight_indices(reference_model)
    num_weight_arrays = len(client_weights[0])

    if reference_global_weights is not None and len(reference_global_weights) != num_weight_arrays:
        raise ValueError("reference_global_weights must match client weight length.")

    aggregated_weights = []

    for weight_index in range(num_weight_arrays):
        if weight_index in batch_norm_indices:
            if reference_global_weights is not None:
                aggregated_weights.append(reference_global_weights[weight_index].copy())
            else:
                aggregated_weights.append(client_weights[0][weight_index].copy())
            continue

        weight_stack = [
            client_weight_list[weight_index] for client_weight_list in client_weights
        ]
        aggregated_weights.append(np.mean(weight_stack, axis=0))

    return aggregated_weights


def preserve_local_batch_norm_weights(
    model: tf.keras.Model,
    incoming_weights: list[np.ndarray],
    local_weights_before: list[np.ndarray],
) -> list[np.ndarray]:
    """Merge global incoming weights while keeping local BatchNorm weights."""
    if len(incoming_weights) != len(local_weights_before):
        raise ValueError("incoming_weights and local_weights_before must match.")

    batch_norm_indices = get_batch_norm_weight_indices(model)
    merged_weights = []

    for weight_index, incoming_weight in enumerate(incoming_weights):
        if weight_index in batch_norm_indices:
            merged_weights.append(local_weights_before[weight_index].copy())
        else:
            merged_weights.append(incoming_weight.copy())

    return merged_weights


class FedBNClient(FederatedClient):
    """Federated client that preserves its local BatchNormalization weights."""

    def set_global_weights(self, global_weights: list[np.ndarray]) -> None:
        """Apply global non-BN weights while preserving local BN weights."""
        if not isinstance(global_weights, list) or not global_weights:
            raise ValueError("global_weights must be a non-empty list.")

        local_weights_before = self.model.get_weights()
        merged_weights = preserve_local_batch_norm_weights(
            self.model,
            global_weights,
            local_weights_before,
        )
        self.model.set_weights(merged_weights)


class FedBNServer(FederatedServer):
    """Federated server using FedBN aggregation instead of standard FedAvg."""

    def aggregate_and_update(
        self,
        client_weights: list[list[np.ndarray]],
    ) -> list[np.ndarray]:
        """Aggregate with FedBN and update the global model."""
        aggregated_weights = aggregate_fedbn_weights(
            client_weights,
            reference_model=self.global_model,
            reference_global_weights=self.global_model.get_weights(),
        )
        self.set_global_weights(aggregated_weights)

        return aggregated_weights


def compare_weight_changes_by_bn_status(
    before_weights: list[np.ndarray],
    after_weights: list[np.ndarray],
    bn_indices: set[int],
) -> dict:
    """Report whether BN and non-BN weights changed between two snapshots."""
    if len(before_weights) != len(after_weights):
        raise ValueError("Weight lists must have the same length.")

    bn_changed = False
    non_bn_changed = False

    for weight_index, (before_weight, after_weight) in enumerate(
        zip(before_weights, after_weights)
    ):
        changed = not np.array_equal(before_weight, after_weight)

        if weight_index in bn_indices and changed:
            bn_changed = True
        elif weight_index not in bn_indices and changed:
            non_bn_changed = True

    return {
        "bn_changed": bn_changed,
        "non_bn_changed": non_bn_changed,
    }


if __name__ == "__main__":
    federated_datasets = create_federated_datasets(batch_size=8)
    clients = [
        FedBNClient("Client_A", federated_datasets["Client_A"]),
        FedBNClient("Client_B", federated_datasets["Client_B"]),
    ]
    server = FedBNServer(clients)

    global_weights_before = server.get_global_weights()
    bn_indices = get_batch_norm_weight_indices(server.global_model)
    non_bn_indices = get_non_batch_norm_weight_indices(server.global_model)

    print(f"total weight arrays: {len(global_weights_before)}")
    print(f"batch norm weight indices count: {len(bn_indices)}")
    print(f"non batch norm weight indices count: {len(non_bn_indices)}")

    server.run_federated_training(
        num_rounds=2,
        steps_per_epoch=1,
        epochs=1,
        verbose=0,
    )

    global_weights_after = server.get_global_weights()
    change_report = compare_weight_changes_by_bn_status(
        global_weights_before,
        global_weights_after,
        bn_indices,
    )

    print(f"bn_changed: {change_report['bn_changed']}")
    print(f"non_bn_changed: {change_report['non_bn_changed']}")

    assert len(bn_indices) > 0
    assert change_report["non_bn_changed"] is True
