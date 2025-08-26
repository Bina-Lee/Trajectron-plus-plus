import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Visualize saved trajectory predictions")
parser.add_argument("--prediction_file", type=str, required=True, help="path to saved predictions json")
parser.add_argument("--index", type=int, default=0, help="trajectory index to visualize")
args = parser.parse_args()

with open(args.prediction_file, 'r') as f:
    data = json.load(f)

if len(data) == 0:
    raise ValueError("No predictions found in file")

idx = max(0, min(args.index, len(data) - 1))
entry = data[idx]

history = np.array(entry['history'])
prediction = np.array(entry['prediction'])
future = np.array(entry.get('future', []))

plt.plot(history[:, 0], history[:, 1], 'k--', label='history')
plt.scatter(history[-1, 0], history[-1, 1], c='g', label='current')
plt.plot(prediction[:, 0], prediction[:, 1], 'r', label='prediction')
if future.size > 0:
    plt.plot(future[:, 0], future[:, 1], 'b--', label='future')
plt.axis('equal')
plt.legend()
plt.show()
