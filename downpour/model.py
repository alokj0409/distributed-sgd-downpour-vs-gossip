"""
model.py — CNN model for CIFAR-10 and tensor serialization helpers.

Defines a simple but effective CNN architecture and provides utilities
to serialize/deserialize PyTorch state dicts and gradients for gRPC transport.
"""

import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class CifarCNN(nn.Module):
    """
    A simple CNN for CIFAR-10 classification.

    Architecture:
        Conv2d(3, 32) -> ReLU -> MaxPool
        Conv2d(32, 64) -> ReLU -> MaxPool
        Conv2d(64, 64) -> ReLU
        FC(64*4*4, 512) -> ReLU -> Dropout
        FC(512, 10)
    """

    def __init__(self):
        super(CifarCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Conv block 1: 32x32 -> 16x16
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        # Conv block 2: 16x16 -> 8x8
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        # Conv block 3: 8x8 -> 4x4
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        # Flatten and classify
        x = x.view(x.size(0), -1)  # Flatten to (batch, 64*4*4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Serialization helpers for gRPC transport
# ---------------------------------------------------------------------------

def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize a PyTorch tensor to raw bytes (float32)."""
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def deserialize_tensor(data: bytes, shape: list) -> torch.Tensor:
    """Deserialize raw bytes back to a PyTorch tensor."""
    import numpy as np
    arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
    return torch.from_numpy(arr.copy())


def serialize_state_dict(state_dict: OrderedDict) -> list:
    """
    Serialize a model state_dict into a list of (name, shape, bytes) tuples.
    Suitable for packing into gRPC TensorData messages.
    """
    result = []
    for name, param in state_dict.items():
        shape = list(param.shape)
        data = serialize_tensor(param)
        result.append((name, shape, data))
    return result


def deserialize_state_dict(tensor_list: list, device: str = "cpu") -> OrderedDict:
    """
    Deserialize a list of (name, shape, bytes) tuples back into a state_dict.

    Args:
        tensor_list: List of (name, shape, data_bytes) tuples
        device: Target device for the tensors
    """
    state_dict = OrderedDict()
    for name, shape, data in tensor_list:
        tensor = deserialize_tensor(data, shape)
        state_dict[name] = tensor.to(device)
    return state_dict


def serialize_gradients(model: nn.Module) -> list:
    """
    Extract and serialize gradients from a model's named parameters.
    Returns a list of (name, shape, bytes) tuples for parameters that have gradients.
    """
    result = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            shape = list(param.grad.shape)
            data = serialize_tensor(param.grad)
            result.append((name, shape, data))
    return result


def deserialize_gradients(tensor_list: list, device: str = "cpu") -> OrderedDict:
    """
    Deserialize a list of (name, shape, bytes) tuples back into a gradient dict.
    Same format as deserialize_state_dict but semantically for gradients.
    """
    return deserialize_state_dict(tensor_list, device)


def get_model() -> CifarCNN:
    """Factory function to create a fresh CifarCNN model."""
    return CifarCNN()
