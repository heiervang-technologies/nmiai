import json
import numpy as np

def print_neighborhood(grid, y, x, radius=3):
    class_map = {0: '.', 1: 'S', 2: 'P', 3: 'R', 4: 'F', 5: 'M', 10: 'O', 11: '-'}
    
    lines = []
    for dy in range(-radius, radius + 1):
        row_str = ""
        for dx in range(-radius, radius + 1):
            ny, nx = y + dy, x + dx
            if 0 <= ny < 40 and 0 <= nx < 40:
                cell = grid[ny][nx]
                char = class_map.get(cell, '?')
                if dy == 0 and dx == 0:
                    row_str += f"[{char}]"
                else:
                    row_str += f" {char} "
            else:
                row_str += "   "
        lines.append(row_str)
    return lines

def main():
    file_path = 'tasks/astar-island/replays/round18_seed0.json'
    with open(file_path) as f:
        data = json.load(f)
    frames = data['frames']

    # the "step" in the targets appears to be "i" in extract_tensor, where phase = i % 4
    # so for Target: Step 2 -> 3, i = 2, meaning prev = frames[i-1] = frames[1]
    targets = [
        (1, 26, 30),
        (2, 10, 29),
        (2, 14, 12),
        (2, 25, 8),
        (2, 30, 11),
        (3, 26, 30),
        (4, 37, 13),
        (5, 2, 7),
        (5, 4, 10),
        (5, 4, 15),
    ]

    for i, y, x in targets:
        print(f"\n--- i = {i} (Phase {i%4}) | Coord (y={y}, x={x}) ---")
        grid1 = frames[i-1]['grid']
        grid2 = frames[i]['grid']
        
        lines1 = print_neighborhood(grid1, y, x, radius=3)
        lines2 = print_neighborhood(grid2, y, x, radius=3)
        
        print("PREV (t)                NEXT (t+1)")
        for l1, l2 in zip(lines1, lines2):
            print(f"{l1}   |   {l2}")

if __name__ == '__main__':
    main()
