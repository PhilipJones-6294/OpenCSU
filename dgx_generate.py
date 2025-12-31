"""
dgx_generate.py
Run this on the DGX to create the training dataset.
Generates 50,000 simulation results.
"""
import torch
import numpy as np
from tqdm import tqdm
from opencsu.core import UnifiedSolver, ModelParams

# CONFIG
NUM_SAMPLES = 50000
BATCH_SIZE = 100 # Adjust based on GPU VRAM (A100 can handle more)
GRID_SIZE = 64   # 64x64 is enough for clinical morphology visualization

def generate_dataset():
    print(f"🚀 Starting Data Generation: {NUM_SAMPLES} samples")
    
    data_inputs = []
    data_targets = []
    
    # We create a random sampler for the physics parameters
    # We focus on the variables the doctor will tweak
    
    batches = NUM_SAMPLES // BATCH_SIZE
    
    for _ in tqdm(range(batches)):
        # 1. Randomize Parameters (Batch)
        # mu_T: 0.1 to 1.2 (Covers Circular -> Annular -> Dot)
        mu_t = torch.rand(BATCH_SIZE, device='cuda') * 1.1 + 0.1
        
        # gamma_M: 0.5 to 2.5 (Feedback Strength)
        gamma_m = torch.rand(BATCH_SIZE, device='cuda') * 2.0 + 0.5
        
        # gamma_T_mod: 0.0 to 1.0 (Drug Effect: Antihistamine)
        drug_eff = torch.rand(BATCH_SIZE, device='cuda')
        
        # 2. Run Solvers (Batched)
        # Note: We need a custom batch-solver or just loop fast.
        # Since UnifiedSolver is single-instance, we loop the batch here.
        # On DGX, we can parallelize this using torch.vmap or multiprocessing.
        # For simplicity/compatibility, we run a tight loop.
        
        batch_inputs = torch.stack([mu_t, gamma_m, drug_eff], dim=1).cpu()
        batch_images = []
        
        for i in range(BATCH_SIZE):
            p = ModelParams()
            p.mu_T = float(mu_t[i])
            p.gamma_M = float(gamma_m[i])
            p.gamma_T_mod = float(drug_eff[i])
            
            solver = UnifiedSolver(p, grid_size=GRID_SIZE, device='cuda')
            
            # Fast-forward to steady state
            # We skip intermediate steps to save time
            for _ in range(300): 
                solver.step()
                
            # Capture the visual state (Histamine channel)
            img, _ = solver.get_visual_layers() # Returns numpy
            batch_images.append(torch.tensor(img))
            
        # Stack
        data_inputs.append(batch_inputs)
        data_targets.append(torch.stack(batch_images))
        
    # Save to disk
    print("💾 Saving dataset...")
    X = torch.cat(data_inputs)
    Y = torch.cat(data_targets)
    
    torch.save({'X': X, 'Y': Y}, 'csu_training_data.pt')
    print("✅ Done. Saved to csu_training_data.pt")

if __name__ == "__main__":
    generate_dataset()