"""
opencsu/pharma/patient.py
"""
from dataclasses import dataclass

@dataclass
class PatientProfile:
    age_years: float
    weight_kg: float
    sex: str = 'F'
    liver_function: float = 1.0 
    
    def get_volume_of_distribution(self):
        # Children have higher body water % than adults
        v_per_kg = 0.7 if self.age_years < 12 else 0.6
        if self.sex == 'F': v_per_kg *= 0.95 
        return self.weight_kg * v_per_kg

    def get_metabolic_capacity(self):
        return self.liver_function