"""
Generate sample datasets for Drug Discovery Project
Creates CSV, JSON, and image data
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random


def generate_drug_csv_data(n_samples: int = 10000, output_path: str = "data/raw/drug_data.csv"):
    """Generate CSV dataset with drug properties and target"""
    print(f"Generating {n_samples} drug samples for CSV...")
    
    np.random.seed(42)
    
    # Generate drug IDs
    drug_ids = [f"DRUG_{str(i).zfill(6)}" for i in range(n_samples)]
    
    # Molecular properties
    molecular_weight = np.random.normal(350, 100, n_samples).clip(50, 800)
    logP = np.random.normal(2.5, 1.5, n_samples).clip(-2, 8)  # Lipophilicity
    h_bond_donors = np.random.poisson(2, n_samples).clip(0, 10)
    h_bond_acceptors = np.random.poisson(4, n_samples).clip(0, 15)
    rotatable_bonds = np.random.poisson(5, n_samples).clip(0, 20)
    
    # Pharmacokinetic properties
    bioavailability = np.random.beta(2, 2, n_samples) * 100
    half_life = np.random.exponential(8, n_samples).clip(0.5, 50)  # hours
    clearance = np.random.gamma(2, 2, n_samples).clip(0.1, 30)
    
    # Toxicity indicators
    hepatotoxicity_score = np.random.beta(2, 5, n_samples) * 10
    cardiotoxicity_score = np.random.beta(2, 5, n_samples) * 10
    
    # Chemical properties
    solubility = np.random.normal(3, 1.5, n_samples).clip(-5, 8)  # log scale
    melting_point = np.random.normal(180, 40, n_samples).clip(50, 350)
    pKa = np.random.normal(7, 2, n_samples).clip(2, 14)
    
    # Drug categories
    categories = ['Antibiotic', 'Antiviral', 'Anticancer', 'Cardiovascular', 
                  'Neurological', 'Anti-inflammatory', 'Antidiabetic', 'Immunosuppressant']
    drug_category = np.random.choice(categories, n_samples)
    
    # Development stage
    stages = ['Discovery', 'Preclinical', 'Phase I', 'Phase II', 'Phase III', 'Approved']
    development_stage = np.random.choice(stages, n_samples, p=[0.1, 0.15, 0.2, 0.25, 0.2, 0.1])
    
    # Efficacy scores (correlated with properties)
    efficacy_base = (
        0.3 * (molecular_weight - 350) / 100 +
        0.2 * (bioavailability / 100) +
        0.15 * (5 - np.abs(logP - 2.5)) +
        -0.25 * (hepatotoxicity_score / 10) +
        -0.2 * (cardiotoxicity_score / 10) +
        0.1 * (solubility / 8) +
        np.random.normal(0, 0.5, n_samples)
    )
    efficacy_score = (1 / (1 + np.exp(-efficacy_base)) * 10).clip(1, 10)  # Sigmoid transform to 1-10
    
    # Safety scores (inversely correlated with toxicity)
    safety_score = (10 - (hepatotoxicity_score + cardiotoxicity_score) / 2 + 
                   np.random.normal(0, 1, n_samples)).clip(1, 10)
    
    # Add some missing values (5-10% per column)
    def add_missing(arr, missing_rate=0.07):
        mask = np.random.random(len(arr)) < missing_rate
        arr = arr.copy()
        arr[mask] = np.nan
        return arr
    
    # Target: Drug effectiveness (1 = effective, 0 = not effective)
    # Based on efficacy and safety scores
    effectiveness_prob = (efficacy_score / 10 * 0.6 + safety_score / 10 * 0.4)
    target = (np.random.random(n_samples) < effectiveness_prob).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'drug_id': drug_ids,
        'molecular_weight': add_missing(molecular_weight),
        'logP': add_missing(logP),
        'h_bond_donors': add_missing(h_bond_donors),
        'h_bond_acceptors': add_missing(h_bond_acceptors),
        'rotatable_bonds': add_missing(rotatable_bonds),
        'bioavailability': add_missing(bioavailability),
        'half_life': add_missing(half_life),
        'clearance': add_missing(clearance),
        'hepatotoxicity_score': add_missing(hepatotoxicity_score),
        'cardiotoxicity_score': add_missing(cardiotoxicity_score),
        'solubility': add_missing(solubility),
        'melting_point': add_missing(melting_point),
        'pKa': add_missing(pKa),
        'efficacy_score': efficacy_score,
        'safety_score': safety_score,
        'drug_category': drug_category,
        'development_stage': development_stage,
        'target': target
    })
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Saved {n_samples} samples to {output_path}")
    print(f"  Target distribution: {np.sum(target)} effective ({np.sum(target)/len(target)*100:.1f}%), "
          f"{len(target) - np.sum(target)} not effective ({(len(target) - np.sum(target))/len(target)*100:.1f}%)")
    
    return df


def generate_drug_json_data(n_samples: int = 10000, output_path: str = "data/raw/drug_interactions.json"):
    """Generate JSON dataset with drug interactions and metadata"""
    print(f"Generating {n_samples} drug interaction records for JSON...")
    
    np.random.seed(42)
    random.seed(42)
    
    # Generate drug interaction data
    data = []
    
    protein_targets = ['EGFR', 'VEGFR', 'BCR-ABL', 'HER2', 'PDGFR', 'mTOR', 'CDK4/6', 
                       'PD-1', 'CTLA-4', 'JAK2', 'BRAF', 'MEK', 'PI3K', 'ALK']
    
    mechanisms = ['Competitive inhibition', 'Non-competitive inhibition', 'Allosteric modulation',
                 'Receptor antagonist', 'Enzyme inhibitor', 'Channel blocker', 'Protein synthesis inhibition']
    
    for i in range(n_samples):
        drug_id = f"DRUG_{str(i).zfill(6)}"
        
        # Random interactions
        n_interactions = random.randint(1, 5)
        interactions = random.sample(protein_targets, n_interactions)
        
        # Mechanism of action
        mechanism = random.choice(mechanisms)
        
        # Side effects
        all_side_effects = ['Nausea', 'Headache', 'Fatigue', 'Dizziness', 'Diarrhea', 
                           'Insomnia', 'Rash', 'Hypertension', 'Hypotension', 'Anemia']
        n_side_effects = random.randint(0, 5)
        side_effects = random.sample(all_side_effects, n_side_effects)
        
        # Clinical trial data
        trial_patients = random.randint(50, 5000)
        response_rate = random.uniform(0.2, 0.9)
        
        # Additional metadata
        manufacturer = random.choice(['PharmaCorp', 'BioTech Inc', 'MediLabs', 'DrugCo', 'HealthPharma'])
        year_discovered = random.randint(1995, 2024)
        
        record = {
            'drug_id': drug_id,
            'protein_targets': interactions,
            'mechanism_of_action': mechanism,
            'side_effects': side_effects,
            'trial_patients': trial_patients,
            'response_rate': round(response_rate, 3),
            'manufacturer': manufacturer,
            'year_discovered': year_discovered,
            'num_interactions': len(interactions),
            'num_side_effects': len(side_effects)
        }
        
        data.append(record)
    
    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved {n_samples} interaction records to {output_path}")
    
    return data


def generate_molecular_images(n_images: int = 500, output_dir: str = "data/images"):
    """Generate synthetic molecular structure images"""
    print(f"Generating {n_images} molecular structure images...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    random.seed(42)
    
    for i in range(n_images):
        drug_id = f"DRUG_{str(i).zfill(6)}"
        
        # Create image
        img_size = (128, 128)
        img = Image.new('RGB', img_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Random molecular structure (simplified representation)
        n_nodes = random.randint(8, 20)
        nodes = [(random.randint(10, 118), random.randint(10, 118)) for _ in range(n_nodes)]
        
        # Draw bonds (edges between nodes)
        for j in range(n_nodes - 1):
            # Connect some nodes
            if random.random() > 0.3:
                draw.line([nodes[j], nodes[j+1]], fill='black', width=2)
        
        # Add some ring structures
        n_rings = random.randint(1, 3)
        for _ in range(n_rings):
            center = (random.randint(30, 98), random.randint(30, 98))
            radius = random.randint(10, 20)
            draw.ellipse([center[0]-radius, center[1]-radius, 
                         center[0]+radius, center[1]+radius], 
                        outline='blue', width=2)
        
        # Draw nodes (atoms)
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for node in nodes:
            color = random.choice(colors)
            draw.ellipse([node[0]-3, node[1]-3, node[0]+3, node[1]+3], 
                        fill=color, outline='black')
        
        # Add some functional groups
        n_groups = random.randint(2, 5)
        for _ in range(n_groups):
            x, y = random.randint(5, 123), random.randint(5, 123)
            group_type = random.choice(['O', 'N', 'C', 'S'])
            draw.text((x, y), group_type, fill='black')
        
        # Save image
        img_path = output_dir / f"{drug_id}.png"
        img.save(img_path)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_images} images")
    
    print(f"✓ Saved {n_images} images to {output_dir}")


def generate_all_datasets(n_samples: int = 10000, n_images: int = 500):
    """Generate all datasets (CSV, JSON, images)"""
    print("\n" + "="*60)
    print("GENERATING DRUG DISCOVERY DATASETS")
    print("="*60 + "\n")
    
    # Generate CSV data
    df = generate_drug_csv_data(n_samples)
    
    # Generate JSON data
    json_data = generate_drug_json_data(n_samples)
    
    # Generate images
    generate_molecular_images(n_images)
    
    print("\n" + "="*60)
    print("✓ ALL DATASETS GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nDataset Summary:")
    print(f"  - CSV records: {n_samples}")
    print(f"  - JSON records: {n_samples}")
    print(f"  - Molecular images: {n_images}")
    print(f"  - Total features in CSV: {len(df.columns)}")
    print(f"  - Target variable: 'target' (0=not effective, 1=effective)")
    print("="*60 + "\n")


if __name__ == "__main__":
    generate_all_datasets(n_samples=10000, n_images=500)
