"""Central federated server simulation with simple FedAvg aggregation."""

import numpy as np


try:
    from evotrack_ai.federated_client import FederatedClient
    from evotrack_ai.federated_data import create_federated_datasets
    from evotrack_ai.siamese_model import build_siamese_model
except ModuleNotFoundError:
    from federated_client import FederatedClient
    from federated_data import create_federated_datasets
    from siamese_model import build_siamese_model


def validate_weight_lists(client_weights: list[list[np.ndarray]]) -> None:
    """Validate client model weights before federated aggregation.

    Args:
        client_weights: List containing one weight list per client.

    Raises:
        ValueError: If the weight structure, lengths, or shapes are invalid.
    """
    if not isinstance(client_weights, list) or not client_weights:
        raise ValueError("client_weights must be a non-empty list.")

    first_client_weights = client_weights[0]

    if not isinstance(first_client_weights, list) or not first_client_weights:
        raise ValueError("Each client weight entry must be a non-empty list.")

    expected_num_arrays = len(first_client_weights)
    expected_shapes = []

    for weight_array in first_client_weights:
        if not isinstance(weight_array, np.ndarray):
            raise ValueError("Each weight must be a NumPy array.")

        expected_shapes.append(weight_array.shape)

    for client_index, weights in enumerate(client_weights):
        if not isinstance(weights, list) or not weights:
            raise ValueError(
                f"Client {client_index} weights must be a non-empty list."
            )

        if len(weights) != expected_num_arrays:
            raise ValueError("All clients must have the same number of weights.")

        for weight_index, weight_array in enumerate(weights):
            if not isinstance(weight_array, np.ndarray):
                raise ValueError("Each weight must be a NumPy array.")

            if weight_array.shape != expected_shapes[weight_index]:
                raise ValueError(
                    "All clients must have matching weight shapes. "
                    f"Mismatch at weight index {weight_index}."
                )


def aggregate_weights(client_weights: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Aggregate client weights with simple FedAvg.

    Args:
        client_weights: List containing one weight list per client.

    Returns:
        A new list of averaged NumPy weight arrays.
    """
    validate_weight_lists(client_weights)

    aggregated_weights = []
    num_weight_arrays = len(client_weights[0])

    for weight_index in range(num_weight_arrays):
        weight_stack = [
            client_weight_list[weight_index] for client_weight_list in client_weights
        ]
        averaged_weight = np.mean(weight_stack, axis=0)
        aggregated_weights.append(averaged_weight)

    return aggregated_weights


class FederatedServer:
    """Coordinate federated clients and maintain a global model."""

    def __init__(
        self,
        clients: list,
        learning_rate: float = 1e-4,
    ) -> None:
        """Initialize the federated server.

        Args:
            clients: Non-empty list of federated clients.
            learning_rate: Learning rate used to compile the global model.

        Raises:
            ValueError: If the clients list is invalid.
        """
        if not isinstance(clients, list) or not clients:
            raise ValueError("clients must be a non-empty list.")

        self.clients = clients
        self.global_model = build_siamese_model(learning_rate=learning_rate)

    def get_global_weights(self) -> list[np.ndarray]:
        """Return the current global model weights."""
        return self.global_model.get_weights()

    def set_global_weights(self, weights: list[np.ndarray]) -> None:
        """Update the global model weights."""
        self.global_model.set_weights(weights)

    def distribute_global_weights(self) -> None:
        """Send global weights to all clients without sharing data."""
        global_weights = self.get_global_weights()

        for client in self.clients:
            client.set_global_weights(global_weights)

    def train_clients_one_round(
        self,
        steps_per_epoch: int = 2,
        epochs: int = 1,
        verbose: int = 0,
    ) -> list[list[np.ndarray]]:
        """Train all clients once and collect their local weights."""
        all_client_weights = []

        for client in self.clients:
            client.train_local_epoch(
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                verbose=verbose,
            )
            all_client_weights.append(client.get_local_weights())

        return all_client_weights

    def aggregate_and_update(
        self,
        client_weights: list[list[np.ndarray]],
    ) -> list[np.ndarray]:
        """Aggregate client weights and update the global model."""
        aggregated_weights = aggregate_weights(client_weights)
        self.set_global_weights(aggregated_weights)

        return aggregated_weights

    def run_federated_training(
        self,
        num_rounds: int = 2,
        steps_per_epoch: int = 2,
        epochs: int = 1,
        verbose: int = 0,
    ) -> None:
        """Run several rounds of simple FedAvg federated training."""
        if num_rounds <= 0:
            raise ValueError("num_rounds must be positive.")

        for round_index in range(1, num_rounds + 1):
            print(f"Round {round_index}/{num_rounds}")
            self.distribute_global_weights()
            client_weights = self.train_clients_one_round(
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                verbose=verbose,
            )
            aggregated_weights = self.aggregate_and_update(client_weights)

            print(f"number of clients: {len(self.clients)}")
            print(f"number of weight arrays received: {len(client_weights[0])}")
            print(f"first weight array shape: {aggregated_weights[0].shape}")


if __name__ == "__main__":
    federated_datasets = create_federated_datasets(batch_size=8)
    clients = [
        FederatedClient("Client_A", federated_datasets["Client_A"]),
        FederatedClient("Client_B", federated_datasets["Client_B"]),
    ]
    server = FederatedServer(clients)

    server.run_federated_training(
        num_rounds=2,
        steps_per_epoch=1,
        epochs=1,
        verbose=0,
    )

    global_weights = server.get_global_weights()

    print("federated training completed")
    print(f"global weights count: {len(global_weights)}")
    print(f"first global weight shape: {global_weights[0].shape}")

    assert len(global_weights) > 0
    assert isinstance(global_weights[0], np.ndarray)
