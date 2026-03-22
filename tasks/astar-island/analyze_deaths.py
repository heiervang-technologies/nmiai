import json
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors

with open('replays/dense_training/round17_seed0_sim1.json', 'r') as f:
    data = json.load(f)

for frame in data['frames']:
    step = frame['step']
    if step > 50: break
    settlements = frame['settlements']
    alive = [s for s in settlements if s['alive']]
    print(f"Step {step:03d} - Alive settlements: {len(alive)}")
