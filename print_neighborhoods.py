import json

filepath = "/home/me/ht/nmiai/tasks/astar-island/replays/dense_training/round17_seed0_sim1.json"
with open(filepath, 'r') as f:
    data = json.load(f)

frames = data['frames']

def print_neighborhood(step, r, c, size=3):
    print(f"\n--- Step {step}, Ghost Seed at ({r}, {c}) ---")
    prev_grid = frames[step-1]['grid']
    curr_grid = frames[step]['grid']
    
    half = size // 2
    r_start = max(0, r - half)
    r_end = min(data['height'], r + half + 1)
    c_start = max(0, c - half)
    c_end = min(data['width'], c + half + 1)
    
    print("Previous Grid:")
    for i in range(r_start, r_end):
        row = []
        for j in range(c_start, c_end):
            val = prev_grid[i][j]
            if val == 1: row.append(" S ")
            elif val == 4: row.append(" F ")
            elif val == 10: row.append(" O ")
            else: row.append(f"{val:2d} ")
        print("".join(row))
        
    print("Current Grid:")
    for i in range(r_start, r_end):
        row = []
        for j in range(c_start, c_end):
            if i == r and j == c: row.append("[S]")
            else:
                val = curr_grid[i][j]
                if val == 1: row.append(" S ")
                elif val == 4: row.append(" F ")
                elif val == 10: row.append(" O ")
                else: row.append(f"{val:2d} ")
        print("".join(row))

print_neighborhood(2, 10, 34, 7)
print_neighborhood(6, 3, 28, 7)
print_neighborhood(12, 36, 20, 9)
print_neighborhood(4, 5, 8, 7)
print_neighborhood(8, 13, 7, 7)
