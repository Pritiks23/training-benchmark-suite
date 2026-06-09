import matplotlib.pyplot as plt
import json
import re

def load_log(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            try:
                # extract JSON-like dict from line
                match = re.search(r"\{.*\}", line)
                if match:
                    data.append(json.loads(match.group()))
            except:
                pass
    return data

def extract(data, key):
    return [d[key] for d in data if key in d]

def plot_all():
    logs = {
        "torch": load_log("results/torch.log"),
        "fsdp": load_log("results/fsdp.log"),
        "deepspeed": load_log("results/deepspeed.log"),
    }

    # --- 1. Step Time ---
    plt.figure()
    for name, data in logs.items():
        if "time_ms" in (data[0] if data else {}):
            plt.plot(extract(data, "time_ms"), label=name)

    plt.title("Step Time Comparison")
    plt.xlabel("Step")
    plt.ylabel("ms")
    plt.legend()
    plt.show()

    # --- 2. GPU Utilization ---
    plt.figure()
    for name, data in logs.items():
        if "gpu" in (data[0] if data else {}):
            plt.plot(extract(data, "gpu"), label=name)

    plt.title("GPU Utilization")
    plt.xlabel("Step")
    plt.ylabel("%")
    plt.legend()
    plt.show()

    # --- 3. Memory Usage ---
    plt.figure()
    for name, data in logs.items():
        if "mem" in (data[0] if data else {}):
            plt.plot(extract(data, "mem"), label=name)

    plt.title("GPU Memory Usage")
    plt.xlabel("Step")
    plt.ylabel("MB")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_all()
