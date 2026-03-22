import torch

def interpolate_global_tensor(global_tensor, scalar):
    """
    Applies a continuous regime scalar to the global transition tensor.
    Scales the probabilities of transitioning to Settle (1) and Port (2) 
    by the scalar, and renormalizes the remaining classes.
    
    Args:
        global_tensor: torch.Tensor of shape (N, 12) containing base probabilities
        scalar: float, the continuous regime multiplier (e.g., 1.15)
        
    Returns:
        scaled_tensor: torch.Tensor of shape (N, 12)
    """
    scaled = global_tensor.clone()
    
    # Extract original civilized mass (Settle + Port)
    civ_mass = scaled[:, 1] + scaled[:, 2]
    
    # Scale it
    new_civ_mass = civ_mass * scalar
    
    # Prevent over-saturation (must leave at least 1e-6 for other states if they existed)
    max_allowed = torch.ones_like(new_civ_mass) - 1e-6
    new_civ_mass = torch.min(new_civ_mass, max_allowed)
    
    # Calculate the actual applied scalar (in case of clipping)
    # add small epsilon to prevent division by zero
    safe_civ_mass = torch.clamp(civ_mass, min=1e-12)
    actual_scalar = new_civ_mass / safe_civ_mass
    
    # Zero out the scalar for rows that originally had 0 civ mass
    actual_scalar[civ_mass == 0] = 1.0
    
    # Scale Settle and Port
    scaled[:, 1] *= actual_scalar
    scaled[:, 2] *= actual_scalar
    
    # Calculate how much mass is left for the remaining classes
    old_remaining_mass = 1.0 - civ_mass
    new_remaining_mass = 1.0 - new_civ_mass
    
    # Compute the ratio to scale down the remaining classes
    safe_old_remaining = torch.clamp(old_remaining_mass, min=1e-12)
    shrink_ratio = new_remaining_mass / safe_old_remaining
    shrink_ratio[old_remaining_mass == 0] = 1.0 # If no remaining mass originally, ratio doesn't matter
    
    # Apply shrink ratio to all non-civ classes
    mask = torch.ones(12, dtype=torch.bool, device=global_tensor.device)
    mask[1] = False
    mask[2] = False
    
    scaled[:, mask] *= shrink_ratio.unsqueeze(1)
    
    # Ensure perfect normalization due to floating point math
    scaled /= scaled.sum(dim=1, keepdim=True)
    
    return scaled
