"""
opencsu/pharma/drug_db.py
"""
from dataclasses import dataclass

@dataclass
class Drug:
    name: str
    brand_hint: str
    target: str          # H1, H2, TCA, ACE-I, B2-ANT
    potency: float       # Relative binding
    half_life: float
    max_dose_mg: float   # For UI Slider limits
    cyp_inhibitor: bool = False
    immunomodulator: bool = False
    bk_decay_modifier: float = 1.0  # 1.0=Neutral, <1.0=Inhibits Breakdown (BAD)

STANDARD_FORMULARY = {
    'loratadine': Drug('Loratadine', 'Claritin', 'H1', 1.0, 8.0, 20.0),
    'fexofenadine': Drug('Fexofenadine', 'Allegra', 'H1', 1.2, 14.0, 180.0),
    'cetirizine': Drug('Cetirizine', 'Zyrtec', 'H1', 2.5, 8.0, 20.0),
    
    'famotidine': Drug('Famotidine', 'Pepcid', 'H2', 3.0, 3.0, 40.0),
    'cimetidine': Drug('Cimetidine', 'Tagamet', 'H2', 0.8, 2.0, 800.0, cyp_inhibitor=True, immunomodulator=True),
    
    'doxepin': Drug('Doxepin', 'Silenor', 'TCA', 150.0, 17.0, 50.0),
    
    'lisinopril': Drug('Lisinopril', 'Zestril', 'ACE-I', 0.0, 12.0, 40.0, bk_decay_modifier=0.1),
    'icatibant': Drug('Icatibant', 'Firazyr', 'B2-ANT', 100.0, 6.0, 30.0)
}