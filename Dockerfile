FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir grpcio grpcio-tools matplotlib

# Set working directory
WORKDIR /app

# Copy proto definitions
COPY downpour/proto/ /app/downpour/proto/
COPY gossip/proto/ /app/gossip/proto/

# Generate gRPC code for downpour
RUN mkdir -p /app/downpour/generated \
    && python -m grpc_tools.protoc -I /app/downpour/proto --python_out=/app/downpour/generated --grpc_python_out=/app/downpour/generated /app/downpour/proto/parameter_server.proto

# Generate gRPC code for gossip
RUN mkdir -p /app/gossip/generated \
    && python -m grpc_tools.protoc -I /app/gossip/proto --python_out=/app/gossip/generated --grpc_python_out=/app/gossip/generated /app/gossip/proto/gossip_service.proto

# Copy the rest of the code
COPY downpour/ /app/downpour/
COPY gossip/ /app/gossip/

# Create logs directories
RUN mkdir -p /app/downpour/logs /app/gossip/logs

# Set PYTHONPATH
ENV PYTHONPATH=/app
