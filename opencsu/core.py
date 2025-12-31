"""
opencsu/core.py
Unified Solver: Seirin-Lee (Histamine) + Kallikrein-Kinin (Bradykinin)
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelParams:
    # --- SEIRIN-LEE (Histamine/TF) ---
    D_hist: float = 0.1     # u1 Diffusion
    D_coag: float = 0.05    # u4 Diffusion
    
    delta_M: float = 0.01   # Basal Mast
    delta_T: float = 0.01   # Basal TF
    delta_B: float = 0.01   # Basal Basophil
    
    mu_M: float = 0.5       # Histamine Decay
    mu_T: float = 0.8       # TF Decay (0.8=Annular, 0.2=Circular)
    mu_B: float = 0.5       # Basophil Decay
    mu_C: float = 0.5       # Coag Decay
    
    gamma_M: float = 1.5    # Feedback: Coag -> Mast
    gamma_T: float = 1.8    # Histamine -> TF
    gamma_B: float = 1.2    # TF -> Basophil
    gamma_C: float = 2.0    # TF -> Coag
    
    alpha: float = 5.0      # Adenosine Inhibition
    u200: float = 0.67      # Gap Threshold
    beta: float = 50.0      # Switch Steepness

    # --- BRADYKININ SYSTEM ---
    D_bk: float = 0.15      # u6 Diffusion (Fast)
    
    phi_tryptase: float = 0.5 # CROSS-TALK: Mast(u1) -> Kallikrein(u5)
    delta_Kal: float = 0.05   # Basal Kallikrein activation
    gamma_BK: float = 2.0     # Kal -> BK production
    
    mu_Kal: float = 0.8       # C1-INH clearance of Kallikrein
    mu_BK: float = 2.0        # ACE clearance of Bradykinin (Target of Lisinopril)
    
    # --- PHARMA MODIFIERS (Runtime Dynamic) ---
    gamma_T_mod: float = 1.0       # H1 Blockade effect (0.0 - 1.0)
    gamma_M_mod: float = 1.0       # H2 Blockade effect
    mu_T_boost: float = 1.0        # Immunomodulation (Tagamet)
    mu_BK_mod: float = 1.0         # ACE Inhibitor effect (Reduces decay!)
    b2_block: float = 0.0          # Icatibant effect

class UnifiedSolver(torch.nn.Module):
    def __init__(self, params: ModelParams, grid_size=256, dt=0.01, device='cuda'):
        super().__init__()
        self.p = params
        self.dt = dt
        self.grid_size = grid_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # STATE VECTOR: 6 Channels
        # 0: Mast Cell Histamine (u1)
        # 1: Tissue Factor (u2)
        # 2: Basophil Histamine (u3)
        # 3: Coagulation Factors (u4)
        # 4: Plasma Kallikrein (u5)
        # 5: Bradykinin (u6)
        self.state = torch.zeros(6, grid_size, grid_size, device=self.device)
        self.initialize()

    def initialize(self):
        # Create a localized trigger
        y, x = torch.meshgrid(torch.arange(self.grid_size), torch.arange(self.grid_size), indexing='ij')
        center = self.grid_size // 2
        r2 = ((x - center)**2 + (y - center)**2).to(self.device).float()
        
        # Initial Basophil Activation (u3)
        self.state[2] = 2.0 * torch.exp(-r2 / 50.0)
        # Slight noise in TF (u1) and Kallikrein (u5) for symmetry breaking
        self.state[1] += 0.01 * torch.rand_like(self.state[1])
        self.state[4] += 0.01 * torch.rand_like(self.state[4])

    def reaction_dynamics(self, u):
        p = self.p
        u1, u2, u3, u4, u5, u6 = u[0], u[1], u[2], u[3], u[4], u[5]
        
        # --- PATHWAY A: HISTAMINE (Seirin-Lee) ---
        inh_M = 1.0 / (1.0 + p.alpha * u1**2 / (1.0 + u1**2))
        inh_T = 1.0 / (1.0 + p.alpha * (u1+u3)**2 / (1.0 + (u1+u3)**2))
        switch = 1.0 / (1.0 + torch.exp(-p.beta * (u2 - p.u200)))
        
        # Effective Parameters (Pharma Modified)
        g_T = p.gamma_T * p.gamma_T_mod
        g_M = p.gamma_M * p.gamma_M_mod
        m_T = p.mu_T * p.mu_T_boost
        
        du1 = p.delta_M + g_M * u4 * inh_M - p.mu_M * u1
        du2 = p.delta_T + g_T * ((u1+u3)/(1+(u1+u3))) * inh_T - m_T * u2
        du3 = p.delta_B + p.gamma_B * u2 - p.mu_B * u3
        du4 = p.gamma_C * switch - p.mu_C * u4
        
        # --- PATHWAY B: BRADYKININ (Kallikrein-Kinin) ---
        # Activation: FXII (linked to u4) + Tryptase (from u1 Mast Cells)
        kal_source = p.delta_Kal * (u4 + p.phi_tryptase * u1)
        du5 = kal_source - p.mu_Kal * u5
        
        # Bradykinin: Produced by u5, Decayed by ACE (mu_BK_mod affects this)
        bk_decay = p.mu_BK * p.mu_BK_mod 
        du6 = p.gamma_BK * u5 - bk_decay * u6
        
        return torch.stack([du1, du2, du3, du4, du5, du6])

    def step(self):
        # RK4 Integration
        k1 = self.reaction_dynamics(self.state)
        k2 = self.reaction_dynamics(self.state + 0.5*self.dt*k1)
        k3 = self.reaction_dynamics(self.state + 0.5*self.dt*k2)
        k4 = self.reaction_dynamics(self.state + self.dt*k3)
        self.state += (self.dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        
        # Diffusion (Histamine, Coag, Bradykinin)
        for idx, D in [(0, self.p.D_hist), (3, self.p.D_coag), (5, self.p.D_bk)]:
            u = self.state[idx]
            lap = torch.roll(u,1,0) + torch.roll(u,-1,0) + torch.roll(u,1,1) + torch.roll(u,-1,1) - 4*u
            self.state[idx] += D * lap * self.dt
            
        self.state = torch.clamp(self.state, min=0.0)

    def get_visual_layers(self):
        """Returns separate channels for visualization"""
        # Channel 1: Histamine Wheal (TF-driven gap formation)
        wheal = 1.0 / (1.0 + torch.exp(-self.p.beta * (self.state[1] - self.p.u200)))
        
        # Channel 2: Angioedema (Bradykinin-driven swelling)
        u6 = self.state[5]
        angio = (u6**2 / (1.0 + u6**2)) * (1.0 - self.p.b2_block)
        
        return wheal.cpu().numpy(), angio.cpu().numpy()