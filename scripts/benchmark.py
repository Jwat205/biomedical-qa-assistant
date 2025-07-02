import time, statistics
import httpx

url = "http://localhost:8000/{endpoint}"
client = httpx.Client()

latencies = []
for _ in range(500):
    start = time.perf_counter()
    r = client.get(url)
    r.raise_for_status()
    latencies.append((time.perf_counter() - start) * 1000)  # ms

print(f"avg: {statistics.mean(latencies):.1f} ms")
print(f"p95: {statistics.quantiles(latencies, n=100)[94]:.1f} ms")
print(f"max: {max(latencies):.1f} ms")
