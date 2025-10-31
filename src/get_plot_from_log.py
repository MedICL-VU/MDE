import re
import matplotlib.pyplot as plt

log_path = "/home/hao/hao/2025_june/MDE/src/checkpoints_hackathon/mde_518/logs/train/training_2025-06-22_16-29-58.log"
epochs = []
rmses = []

with open(log_path, "r") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    # Look for summary line with 8+ comma-separated floats
    floatline = re.match(r".*?([\-0-9\.]+,\s+){8,}[\-0-9\.]+", line)
    if floatline:
        # RMSE is 6th value (index 5)
        floats = [float(x.strip()) for x in re.findall(r"-?\d+\.\d+", line)]
        if len(floats) >= 6:
            rmse = floats[5]
            rmses.append(rmse)
            # Look for epoch number within next 5 lines
            for j in range(idx, min(idx + 10, len(lines))):
                match = re.search(r"Epoch:\s*(\d+)", lines[j])
                if match:
                    epoch = int(match.group(1))
                    epochs.append(epoch)
                    break
            else:
                # No epoch found, just count by order
                epochs.append(len(epochs) + 1)
            print(f"Found: epoch={epochs[-1]}, rmse={rmse}")

# Plot
if epochs and rmses:
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, rmses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("RMSE per Epoch (parsed from summary blocks)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No RMSE values found.")
