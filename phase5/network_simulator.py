"""
Cross-platform gRPC Network Simulator Interceptor.
Simulates network latency (ms) and bandwidth limits (Mbps) for gRPC calls
without requiring Linux kernel modules (tc/netem). Works 100% offline.
"""

import time
import grpc


class NetworkSimulatorClientInterceptor(grpc.UnaryUnaryClientInterceptor):
    """
    gRPC Client Interceptor that injects artificial latency and bandwidth constraints.
    """

    def __init__(self, latency_ms=0.0, bandwidth_mbps=0.0):
        """
        :param latency_ms: One-way network latency in milliseconds.
        :param bandwidth_mbps: Bandwidth limit in Megabits per second (0.0 = unlimited).
        """
        self.latency_ms = float(latency_ms)
        self.bandwidth_mbps = float(bandwidth_mbps)

    def intercept_unary_unary(self, continuation, client_call_details, request):
        # 1. Measure outbound request payload size
        request_bytes = request.ByteSize() if hasattr(request, "ByteSize") else 0

        # 2. Inject artificial one-way latency delay
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

        # 3. Inject bandwidth throttling delay for outbound payload
        if self.bandwidth_mbps > 0 and request_bytes > 0:
            # bits / bits_per_second
            bandwidth_bps = self.bandwidth_mbps * 1e6
            tx_delay = (request_bytes * 8.0) / bandwidth_bps
            time.sleep(tx_delay)

        start_time = time.time()
        response = continuation(client_call_details, request)
        response_bytes = response.ByteSize() if hasattr(response, "ByteSize") else 0

        # 4. Inject artificial return latency delay
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

        # 5. Inject bandwidth throttling delay for inbound response payload
        if self.bandwidth_mbps > 0 and response_bytes > 0:
            bandwidth_bps = self.bandwidth_mbps * 1e6
            rx_delay = (response_bytes * 8.0) / bandwidth_bps
            time.sleep(rx_delay)

        return response


def create_simulated_channel(target, latency_ms=0.0, bandwidth_mbps=0.0, options=None):
    """
    Helper function to create an intercepted gRPC channel with latency & bandwidth simulation.
    """
    if options is None:
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
    
    channel = grpc.insecure_channel(target, options=options)
    
    if latency_ms > 0 or bandwidth_mbps > 0:
        interceptor = NetworkSimulatorClientInterceptor(latency_ms, bandwidth_mbps)
        return grpc.intercept_channel(channel, interceptor)
    
    return channel
