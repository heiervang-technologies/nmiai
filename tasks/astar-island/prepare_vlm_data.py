"""Prepare Astar Island ground truth data for VLM few-shot learning.

Creates text and image representations for each ground truth file.
"""

import json
import os
import numpy as np
from pathlib import Path

# Try importing matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# === MAPPINGS ===

# Initial grid codes -> char
INITIAL_CHAR = {
    0: '.',   # Empty
    1: 'S',   # Settlement
    2: 'P',   # Port
    4: 'F',   # Forest
    5: 'M',   # Mountain
    10: 'O',  # Ocean
    11: '.',  # Plains (same as empty visually)
}

# Ground truth argmax index -> char
GT_CHAR = {
    0: '.',   # Empty
    1: 'S',   # Settlement
    2: 'P',   # Port
    3: 'R',   # Ruin
    4: 'F',   # Forest
    5: 'M',   # Mountain
}

# Ground truth argmax index -> name
GT_NAME = {
    0: 'Empty',
    1: 'Settlement',
    2: 'Port',
    3: 'Ruin',
    4: 'Forest',
    5: 'Mountain',
}

# Initial grid code -> name
INITIAL_NAME = {
    0: 'Empty',
    1: 'Settlement',
    2: 'Port',
    4: 'Forest',
    5: 'Mountain',
    10: 'Ocean',
    11: 'Plains',
}

# Color maps (RGB tuples 0-1)
INITIAL_COLORS = {
    0:  (0.784, 0.722, 0.541),  # Empty - sand
    1:  (0.831, 0.463, 0.039),  # Settlement - orange
    2:  (0.055, 0.478, 0.565),  # Port - teal
    4:  (0.176, 0.353, 0.153),  # Forest - green
    5:  (0.420, 0.447, 0.498),  # Mountain - gray
    10: (0.118, 0.227, 0.373),  # Ocean - dark blue
    11: (0.784, 0.722, 0.541),  # Plains - sand (same as empty)
}

GT_COLORS = {
    0: (0.784, 0.722, 0.541),  # Empty - sand #c8b88a
    1: (0.831, 0.463, 0.039),  # Settlement - orange #d4760a
    2: (0.055, 0.478, 0.565),  # Port - teal #0e7490
    3: (0.498, 0.114, 0.114),  # Ruin - dark red #7f1d1d
    4: (0.176, 0.353, 0.153),  # Forest - green #2d5a27
    5: (0.420, 0.447, 0.498),  # Mountain - gray #6b7280
}

# Diff colors
DIFF_UNCHANGED = (0.85, 0.85, 0.85)  # light gray
DIFF_COLORS = {
    'to_settlement': (0.831, 0.463, 0.039),
    'to_ruin': (0.498, 0.114, 0.114),
    'to_port': (0.055, 0.478, 0.565),
    'to_forest': (0.176, 0.353, 0.153),
    'to_empty': (1.0, 1.0, 0.6),
    'to_mountain': (0.420, 0.447, 0.498),
}


def load_ground_truth(path):
    with open(path) as f:
        data = json.load(f)
    initial = np.array(data['initial_grid'])
    gt = np.array(data['ground_truth'])
    return initial, gt


def grid_to_image(grid, color_map, scale=10):
    """Convert a 2D grid to an RGB image using a color map."""
    h, w = grid.shape
    img = np.zeros((h, w, 3))
    for code, color in color_map.items():
        mask = grid == code
        img[mask] = color
    # Scale up
    img_scaled = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    return img_scaled


def compute_entropy(gt_probs):
    """Compute entropy for each cell."""
    # gt_probs is 40x40x6
    eps = 1e-10
    p = gt_probs + eps
    p = p / p.sum(axis=2, keepdims=True)
    entropy = -np.sum(p * np.log(p), axis=2)
    return entropy


def make_text_representation(initial, gt, out_path):
    """Create text representation of initial and final grids."""
    argmax = np.argmax(gt, axis=2)
    entropy = compute_entropy(gt)
    high_entropy_threshold = 0.5  # bits

    lines = []

    # Initial grid
    lines.append("=== INITIAL STATE ===")
    lines.append("Legend: O=Ocean .=Plains/Empty S=Settlement P=Port F=Forest M=Mountain")
    lines.append("")
    for row in range(40):
        chars = []
        for col in range(40):
            chars.append(INITIAL_CHAR.get(initial[row, col], '?'))
        lines.append(''.join(chars))

    lines.append("")

    # Summary stats for initial
    lines.append("--- Initial State Stats ---")
    unique, counts = np.unique(initial, return_counts=True)
    for val, cnt in zip(unique, counts):
        name = INITIAL_NAME.get(val, f'Unknown({val})')
        lines.append(f"  {name}: {cnt}")

    lines.append("")
    lines.append("=== GROUND TRUTH (argmax) ===")
    lines.append("Legend: .=Empty S=Settlement P=Port R=Ruin F=Forest M=Mountain")
    lines.append("")
    for row in range(40):
        chars = []
        for col in range(40):
            chars.append(GT_CHAR.get(argmax[row, col], '?'))
        lines.append(''.join(chars))

    lines.append("")

    # Summary stats for ground truth
    lines.append("--- Ground Truth Stats ---")
    unique_gt, counts_gt = np.unique(argmax, return_counts=True)
    for val, cnt in zip(unique_gt, counts_gt):
        name = GT_NAME.get(val, f'Unknown({val})')
        lines.append(f"  {name}: {cnt}")

    lines.append("")

    # High entropy cells
    high_ent = entropy > high_entropy_threshold
    lines.append(f"--- High Entropy Cells (>{high_entropy_threshold} bits): {high_ent.sum()} cells ---")
    lines.append("Overlay (* = high entropy):")
    lines.append("")
    for row in range(40):
        chars = []
        for col in range(40):
            if high_ent[row, col]:
                chars.append('*')
            else:
                chars.append(GT_CHAR.get(argmax[row, col], '?'))
        lines.append(''.join(chars))

    lines.append("")

    # Change summary
    lines.append("--- Changes from Initial to Final ---")
    # Map initial codes to GT-compatible codes
    # Initial: 0=Empty, 1=Settlement, 2=Port, 4=Forest, 5=Mountain, 10=Ocean, 11=Plains
    # GT:      0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain
    # Ocean/Plains in initial -> Empty(0) in GT space
    initial_mapped = np.zeros_like(initial)
    initial_mapped[initial == 0] = 0
    initial_mapped[initial == 1] = 1
    initial_mapped[initial == 2] = 2
    initial_mapped[initial == 4] = 4
    initial_mapped[initial == 5] = 5
    initial_mapped[initial == 10] = 0  # Ocean -> Empty in GT space
    initial_mapped[initial == 11] = 0  # Plains -> Empty in GT space

    changed = initial_mapped != argmax
    lines.append(f"  Total cells changed: {changed.sum()}")

    # Breakdown of changes
    for gt_val in range(6):
        for init_val in [0, 1, 2, 4, 5, 10, 11]:
            init_name = INITIAL_NAME.get(init_val, '?')
            gt_name = GT_NAME.get(gt_val, '?')
            mask = (initial == init_val) & (argmax == gt_val) & changed
            cnt = mask.sum()
            if cnt > 0:
                lines.append(f"  {init_name} -> {gt_name}: {cnt}")

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


def make_images(initial, gt, base_path):
    """Create initial, final, and diff images."""
    argmax = np.argmax(gt, axis=2)
    scale = 10

    # Initial image
    img_initial = grid_to_image(initial, INITIAL_COLORS, scale=scale)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_initial)
    ax.set_title('Initial State', fontsize=12)
    ax.axis('off')
    # Legend
    patches = [Patch(facecolor=INITIAL_COLORS[k], label=INITIAL_NAME[k])
               for k in sorted(INITIAL_COLORS.keys())]
    ax.legend(handles=patches, loc='upper left', fontsize=7, framealpha=0.8)
    plt.tight_layout()
    plt.savefig(f"{base_path}_initial.png", dpi=80, bbox_inches='tight')
    plt.close()

    # Final/GT image
    img_final = grid_to_image(argmax, GT_COLORS, scale=scale)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_final)
    ax.set_title('Ground Truth (argmax)', fontsize=12)
    ax.axis('off')
    patches = [Patch(facecolor=GT_COLORS[k], label=GT_NAME[k])
               for k in sorted(GT_COLORS.keys())]
    ax.legend(handles=patches, loc='upper left', fontsize=7, framealpha=0.8)
    plt.tight_layout()
    plt.savefig(f"{base_path}_final.png", dpi=80, bbox_inches='tight')
    plt.close()

    # Diff image - show what changed
    # Map initial to GT space for comparison
    initial_mapped = np.zeros_like(initial)
    initial_mapped[initial == 0] = 0
    initial_mapped[initial == 1] = 1
    initial_mapped[initial == 2] = 2
    initial_mapped[initial == 4] = 4
    initial_mapped[initial == 5] = 5
    initial_mapped[initial == 10] = 0
    initial_mapped[initial == 11] = 0

    h, w = initial.shape
    diff_img = np.zeros((h, w, 3))
    changed = initial_mapped != argmax

    for r in range(h):
        for c in range(w):
            if not changed[r, c]:
                diff_img[r, c] = DIFF_UNCHANGED
            else:
                # Color by what it became
                new_val = argmax[r, c]
                diff_img[r, c] = GT_COLORS.get(new_val, (1, 0, 1))

    diff_img_scaled = np.repeat(np.repeat(diff_img, scale, axis=0), scale, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(diff_img_scaled)
    ax.set_title('Changes (gray=unchanged)', fontsize=12)
    ax.axis('off')
    # Legend for diff
    diff_patches = [Patch(facecolor=DIFF_UNCHANGED, label='Unchanged')]
    for k in sorted(GT_COLORS.keys()):
        diff_patches.append(Patch(facecolor=GT_COLORS[k], label=f'-> {GT_NAME[k]}'))
    ax.legend(handles=diff_patches, loc='upper left', fontsize=7, framealpha=0.8)
    plt.tight_layout()
    plt.savefig(f"{base_path}_diff.png", dpi=80, bbox_inches='tight')
    plt.close()


def main():
    gt_dir = Path("tasks/astar-island/ground_truth")
    out_dir = Path("tasks/astar-island/vlm_data")
    out_dir.mkdir(exist_ok=True)

    gt_files = sorted(gt_dir.glob("round*_seed*.json"))
    print(f"Found {len(gt_files)} ground truth files")

    for gt_file in gt_files:
        name = gt_file.stem  # e.g. round1_seed0
        print(f"Processing {name}...")

        initial, gt = load_ground_truth(gt_file)

        # Text representation
        text_path = out_dir / f"{name}_text.txt"
        make_text_representation(initial, gt, str(text_path))

        # Image representations
        base_path = str(out_dir / name)
        make_images(initial, gt, base_path)

    print("Done! Creating prompt template...")

    # Create prompt template
    create_prompt_template(out_dir)
    print("All done!")


def create_prompt_template(out_dir):
    """Create a few-shot prompt template for Gemini."""
    template = """You are an expert at predicting how island civilizations evolve over time in the Astar Island simulation.

## Task
Given an initial 40x40 grid map of an island, predict the final state after the simulation runs. The simulation models settlement growth, resource usage, and terrain changes.

## Cell Types
- Empty (code 0): Unoccupied land
- Settlement (code 1): Human settlement
- Port (code 2): Coastal trading port
- Ruin (code 3): Destroyed/abandoned settlement (only appears in output)
- Forest (code 4): Forested area
- Mountain (code 5): Mountainous terrain

## Key Patterns to Learn
- Settlements tend to grow near existing settlements and ports
- Forests can be cleared for settlement expansion
- Some settlements become ruins
- Mountains and ocean don't change
- Ports appear at coast-settlement intersections
- Settlement growth follows proximity rules

## Examples

### Example 1: Round 1, Seed 0
**Initial State:**
[IMAGE: round1_seed0_initial.png]

**Text Grid (Initial):**
[Insert contents of round1_seed0_text.txt - Initial section]

**Result After Simulation:**
[IMAGE: round1_seed0_final.png]

**What Changed:**
[IMAGE: round1_seed0_diff.png]

---

### Example 2: Round 2, Seed 0
**Initial State:**
[IMAGE: round2_seed0_initial.png]

**Text Grid (Initial):**
[Insert contents of round2_seed0_text.txt - Initial section]

**Result After Simulation:**
[IMAGE: round2_seed0_final.png]

**What Changed:**
[IMAGE: round2_seed0_diff.png]

---

### Example 3: Round 3, Seed 0
**Initial State:**
[IMAGE: round3_seed0_initial.png]

**Text Grid (Initial):**
[Insert contents of round3_seed0_text.txt - Initial section]

**Result After Simulation:**
[IMAGE: round3_seed0_final.png]

**What Changed:**
[IMAGE: round3_seed0_diff.png]

---

## Your Turn

Now predict the final state for this new island:

**Initial State:**
[IMAGE: {input_initial_image}]

**Text Grid (Initial):**
{input_text_grid}

**Instructions:**
1. Analyze the initial layout - where are settlements, ports, forests, mountains, and ocean?
2. Based on the patterns from the examples, predict how the civilization will evolve
3. Output a 40x40 grid where each cell is one of: 0 (Empty), 1 (Settlement), 2 (Port), 3 (Ruin), 4 (Forest), 5 (Mountain)
4. Also output a 40x40x6 probability distribution for each cell

Output your prediction as a JSON object with key "prediction" containing a 40x40x6 array of probabilities.
Each cell's 6 values should sum to 1.0 and represent [P(Empty), P(Settlement), P(Port), P(Ruin), P(Forest), P(Mountain)].

```json
{
  "prediction": [[[p0, p1, p2, p3, p4, p5], ...], ...]
}
```
"""

    with open(out_dir / "prompt_template.txt", 'w') as f:
        f.write(template)


if __name__ == "__main__":
    main()
