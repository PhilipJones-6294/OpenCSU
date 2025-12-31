"""
opencsu/pharma/engine.py
"""
from copy import deepcopy
from .drug_db import STANDARD_FORMULARY

def calculate_patient_params(base_params, patient_profile, prescription_mg):
    p = deepcopy(base_params)
    warnings = []
    
    # 1. Patient Stats
    vd = patient_profile.get_volume_of_distribution()
    metabolic_rate = patient_profile.get_metabolic_capacity()
    
    # 2. Check for CYP Inhibitors (Tagamet)
    cyp_inhibition = 0.0
    for name, dose_mg in prescription_mg.items():
        if dose_mg > 0:
            drug = STANDARD_FORMULARY[name]
            if drug.cyp_inhibitor:
                conc = dose_mg / vd
                cyp_inhibition += 0.5 * (conc / (1.0 + conc))

    # 3. Calculate Effects
    h1_occupancy = 0.0
    h2_occupancy = 0.0
    
    for name, dose_mg in prescription_mg.items():
        if dose_mg <= 0: continue
        drug = STANDARD_FORMULARY[name]
        
        # Accumulation Logic
        accumulation_factor = 1.0 / (metabolic_rate * (1.0 - cyp_inhibition * 0.8))
        concentration = (dose_mg / vd) * accumulation_factor
        
        # Warnings
        if drug.name == 'Doxepin' and cyp_inhibition > 0.2:
            warnings.append(f"⛔ DANGER: Doxepin toxicity risk (Level {accumulation_factor:.1f}x)")
        if drug.target == 'ACE-I':
            warnings.append("⚠️ ACE Inhibitor: Angioedema Risk High")
            p.mu_BK_mod *= drug.bk_decay_modifier

        # Binding
        binding = (concentration * drug.potency) / (10.0 + concentration * drug.potency)
        
        if drug.target == 'H1': h1_occupancy += binding
        if drug.target == 'H2': h2_occupancy += binding
        if drug.target == 'TCA': 
            h1_occupancy += binding * 2.0
            h2_occupancy += binding * 2.0
            
        if drug.immunomodulator:
            p.mu_T_boost *= (1.0 + 0.5 * binding)

    # 4. Map to Physics
    p.gamma_T_mod = 1.0 - (0.95 * h1_occupancy)
    p.gamma_M_mod = 1.0 - (0.95 * h2_occupancy)
    
    return p, warnings