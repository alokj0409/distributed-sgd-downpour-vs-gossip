import argparse
import yaml

def generate_downpour(num_workers, output_file):
    services = {
        "parameter-server": {
            "build": {"context": ".", "dockerfile": "Dockerfile"},
            "command": "python /app/downpour/server.py --port 50051",
            "ports": ["50051:50051"],
            "volumes": [
                "./downpour/logs:/app/downpour/logs",
                "./downpour/data:/app/downpour/data"
            ],
            "networks": ["downpour-net"]
        }
    }
    
    for i in range(num_workers):
        services[f"worker-{i}"] = {
            "build": {"context": ".", "dockerfile": "Dockerfile"},
            "command": f"python /app/downpour/worker.py --worker-id {i} --server-address parameter-server:50051 --num-workers {num_workers} --data-dir /app/downpour/data --log-dir /app/downpour/logs",
            "depends_on": ["parameter-server"],
            "volumes": [
                "./downpour/logs:/app/downpour/logs",
                "./downpour/data:/app/downpour/data"
            ],
            "networks": ["downpour-net"]
        }
        
    compose = {
        "services": services,
        "networks": {
            "downpour-net": {"driver": "bridge"}
        }
    }
    
    with open(output_file, "w") as f:
        yaml.dump(compose, f, sort_keys=False)
    print(f"Generated {output_file} with {num_workers} workers.")

def generate_gossip(num_workers, output_file):
    services = {}
    base_port = 50060
    
    for i in range(num_workers):
        peers = [f"worker-{j}:{base_port + j}" for j in range(num_workers) if j != i]
        peers_str = ",".join(peers)
        
        services[f"worker-{i}"] = {
            "build": {"context": ".", "dockerfile": "Dockerfile"},
            "command": f"python /app/gossip/gossip_worker.py --worker-id {i} --num-workers {num_workers} --base-port {base_port} --peers {peers_str} --data-dir /app/gossip/data --log-dir /app/gossip/logs",
            "ports": [f"{base_port + i}:{base_port + i}"],
            "volumes": [
                "./gossip/logs:/app/gossip/logs",
                "./gossip/data:/app/gossip/data"
            ],
            "networks": ["gossip-net"]
        }
        
    compose = {
        "services": services,
        "networks": {
            "gossip-net": {"driver": "bridge"}
        }
    }
    
    with open(output_file, "w") as f:
        yaml.dump(compose, f, sort_keys=False)
    print(f"Generated {output_file} with {num_workers} workers.")

def main():
    parser = argparse.ArgumentParser(description="Generate docker-compose configurations")
    parser.add_argument("--mode", choices=["downpour", "gossip"], required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    
    output = args.output
    if not output:
        output = f"docker-compose.{args.mode}.yml"
        
    if args.mode == "downpour":
        generate_downpour(args.num_workers, output)
    else:
        generate_gossip(args.num_workers, output)

if __name__ == "__main__":
    main()
