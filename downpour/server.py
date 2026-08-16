"""
server.py — Downpour SGD Parameter Server.

Implements a gRPC-based parameter server that:
1. Holds the global model weights
2. Accepts gradient pushes from workers and applies Adagrad updates
3. Serves the latest weights to workers on pull requests
4. Uses threading locks for thread-safe concurrent access
"""

import argparse
import logging
import threading
import time
from concurrent import futures
from collections import OrderedDict

import torch
import grpc

# Import generated protobuf/gRPC modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "generated"))
import parameter_server_pb2 as ps_pb2
import parameter_server_pb2_grpc as ps_grpc

from model import get_model, serialize_state_dict, deserialize_tensor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[SERVER %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ParameterServer")


class AdagradParameterServer(ps_grpc.ParameterServerServicer):
    """
    gRPC servicer implementing the Downpour SGD parameter server.

    Maintains:
        - global_weights: The authoritative model weights (OrderedDict)
        - adagrad_accum: Per-parameter sum of squared gradients for Adagrad
        - global_step: Total number of gradient updates applied
        - lock: Threading lock for safe concurrent access
    """

    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        super().__init__()
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # Initialize global model
        model = get_model()
        self.global_weights = OrderedDict(
            {name: param.clone().detach() for name, param in model.state_dict().items()}
        )

        # Initialize Adagrad accumulator (sum of squared gradients)
        self.adagrad_accum = OrderedDict(
            {name: torch.zeros_like(param) for name, param in self.global_weights.items()}
        )

        self.global_step = 0
        self.lock = threading.Lock()

        # Track worker activity
        self.worker_push_counts = {}

        logger.info(
            f"Parameter server initialized | LR={learning_rate} | "
            f"Parameters: {sum(p.numel() for p in self.global_weights.values()):,}"
        )

    def PushGradients(self, request, context):
        """
        Handle gradient push from a worker.
        Applies Adagrad update: W = W - lr / sqrt(G + eps) * grad
        where G is the accumulated sum of squared gradients.
        """
        worker_id = request.worker_id

        with self.lock:
            for tensor_data in request.gradients:
                name = tensor_data.meta.name
                shape = list(tensor_data.meta.shape)
                grad = deserialize_tensor(tensor_data.data, shape)

                if name in self.global_weights:
                    # Adagrad update
                    self.adagrad_accum[name] += grad ** 2
                    adjusted_lr = self.learning_rate / (
                        torch.sqrt(self.adagrad_accum[name]) + self.epsilon
                    )
                    self.global_weights[name] -= adjusted_lr * grad

            self.global_step += 1

            # Track worker activity
            self.worker_push_counts[worker_id] = (
                self.worker_push_counts.get(worker_id, 0) + 1
            )

        if self.global_step % 100 == 0:
            logger.info(
                f"Global step {self.global_step} | "
                f"Worker pushes: {dict(self.worker_push_counts)}"
            )

        return ps_pb2.PushResponse(
            success=True,
            message=f"Applied gradients at step {self.global_step}",
        )

    def PullWeights(self, request, context):
        """
        Handle weight pull request from a worker.
        Returns the current global model weights.
        """
        worker_id = request.worker_id

        with self.lock:
            serialized = serialize_state_dict(self.global_weights)
            step = self.global_step

        # Build response
        response = ps_pb2.ModelWeights(global_step=step)
        for name, shape, data in serialized:
            tensor_data = ps_pb2.TensorData(
                meta=ps_pb2.TensorMeta(name=name, shape=shape),
                data=data,
            )
            response.weights.append(tensor_data)

        logger.debug(f"Worker {worker_id} pulled weights at step {step}")
        return response

    def get_global_step(self):
        """Thread-safe access to global step."""
        with self.lock:
            return self.global_step

    def get_weights_copy(self):
        """Thread-safe copy of current global weights."""
        with self.lock:
            return OrderedDict(
                {name: param.clone() for name, param in self.global_weights.items()}
            )


def serve(port: int = 50051, learning_rate: float = 0.01, max_workers: int = 10):
    """Start the gRPC parameter server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = AdagradParameterServer(learning_rate=learning_rate)
    ps_grpc.add_ParameterServerServicer_to_server(servicer, server)

    address = f"[::]:{port}"
    server.add_insecure_port(address)
    server.start()
    logger.info(f"Parameter server listening on port {port}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down parameter server...")
        server.stop(grace=5)


def main():
    parser = argparse.ArgumentParser(description="Downpour SGD Parameter Server")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for Adagrad")
    parser.add_argument("--max-workers", type=int, default=10, help="Max gRPC thread pool workers")
    args = parser.parse_args()

    serve(port=args.port, learning_rate=args.lr, max_workers=args.max_workers)


if __name__ == "__main__":
    main()
