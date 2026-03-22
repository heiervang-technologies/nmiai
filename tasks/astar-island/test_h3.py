import json, glob, numpy as np
from scipy.ndimage import convolve

files = glob.glob('/home/me/ht/nmiai/tasks/astar-island/ground_truth/*.json')

def forest_never_grows(ig, gt):
    gt_argmax = np.argmax(gt, axis=-1)
    return (gt_argmax == 4) & (ig != 4)

def mountain_never_grows(ig, gt):
    gt_argmax = np.argmax(gt, axis=-1)
    return (gt_argmax == 5) & (ig != 5)

def no_isolated_civ_in_gt(ig, gt):
    gt_argmax = np.argmax(gt, axis=-1)
    civ_mask = (gt_argmax == 1) | (gt_argmax == 2)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    civ_neighbors = convolve(civ_mask.astype(int), kernel, mode='constant', cval=0)
    return civ_mask & (civ_neighbors == 0)

for h_name, rule in [
    ("Forest Never Grows", forest_never_grows),
    ("Mountain Never Grows", mountain_never_grows),
    ("No Isolated Civ in GT", no_isolated_civ_in_gt),
]:
    violations = 0
    for f in files:
        data = json.load(open(f))
        ig = np.array(data['initial_grid'])
        gt = np.array(data['ground_truth'])
        violations += np.sum(rule(ig, gt))
    print(f"{h_name}: {violations} violations")
