import json
import glob
import numpy as np
from scipy.ndimage import distance_transform_edt, convolve

def test_ca_rule(rule_fn, description="", max_replays=None):
    """
    Exhaustively tests a cellular automaton rule across frame-by-frame replay data.
    
    rule_fn: A function that takes (step, prev_grid, curr_grid, initial_grid)
             and returns a tuple of two boolean numpy arrays of shape (40, 40):
               1. violations_mask: True where the rule was explicitly violated.
               2. applicable_mask: True where the rule's preconditions were met.
             
    This allows us to say: "Out of 50,000 times this situation occurred, the rule held 100% of the time."
    """
    files = glob.glob('tasks/astar-island/replays/*.json')
    if max_replays:
        files = files[:max_replays]
        
    total_applicable = 0
    total_violations = 0
    
    print(f"Testing CA Rule: {description}")
    
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
            
        frames = data['frames']
        if len(frames) < 2:
            continue
            
        initial_grid = np.array(frames[0]['grid'])
        
        for i in range(1, len(frames)):
            prev_grid = np.array(frames[i-1]['grid'])
            curr_grid = np.array(frames[i]['grid'])
            step = frames[i]['step']
            
            violations, applicable = rule_fn(step, prev_grid, curr_grid, initial_grid)
            
            v_count = np.sum(violations)
            a_count = np.sum(applicable)
            
            # Sanity check: can't have a violation where the rule isn't applicable
            # We enforce this strictly for reporting
            v_count = np.sum(violations & applicable)
            
            total_violations += v_count
            total_applicable += a_count
            
    print(f"Result: {total_violations} violations out of {total_applicable} applicable cases.")
    if total_applicable == 0:
        print("⚠️  Rule was NEVER applicable! Check your preconditions.")
    elif total_violations == 0:
        print("✅ Rule holds exhaustively for ALL applicable cases!")
    else:
        print(f"❌ Rule is BROKEN in {total_violations/total_applicable:.2%} of cases.")
    print("-" * 60)


if __name__ == "__main__":
    
    # Example 1: Mountains are static
    def mountains_static_rule(step, prev_grid, curr_grid, initial_grid):
        # Applies to cells that were mountain in the previous step
        applicable = (prev_grid == 5)
        # Violation if they are NOT mountain in the current step
        violation = (curr_grid != 5)
        return violation, applicable
        
    test_ca_rule(mountains_static_rule, "Mountains never change state")
    
    # Example 2: Oceans are static
    def oceans_static_rule(step, prev_grid, curr_grid, initial_grid):
        applicable = (prev_grid == 10)
        violation = (curr_grid != 10)
        return violation, applicable
        
    test_ca_rule(oceans_static_rule, "Oceans never change state")

    # Example 3: Settlements never spawn on Ocean or Mountain
    def no_settlement_on_impassable(step, prev_grid, curr_grid, initial_grid):
        # Applies when a new settlement or port is formed
        applicable = ((curr_grid == 1) | (curr_grid == 2)) & ((prev_grid != 1) & (prev_grid != 2))
        # Violation if it spawned on a mountain or ocean
        violation = (initial_grid == 5) | (initial_grid == 10)
        return violation, applicable
        
    test_ca_rule(no_settlement_on_impassable, "Settlements never grow onto Mountain or Ocean")

    # Example 4: Growth Phase (Step % 4 == 0) is the ONLY time Plains turn to Settlements
    # Note: Through analysis, we found phases. Let's test if growth *strictly* only happens on specific modulo steps
    def strict_growth_phase(step, prev_grid, curr_grid, initial_grid):
        # Applies whenever Plains become Settlement
        applicable = (prev_grid == 11) & (curr_grid == 1)
        # We look at the data: when does this actually happen? 
        # Modulo 4: 0, 1, 2, or 3? Let's assume it only happens on step % 4 == 0
        violation = (step % 4 != 0) 
        return violation, applicable
        
    test_ca_rule(strict_growth_phase, "Plains only become Settlements on Step % 4 == 0")
