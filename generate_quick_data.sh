#!/bin/bash

# Quick Start Script for Drug Discovery AI Project
# This script generates the dataset without requiring all dependencies

echo "=================================================="
echo "Drug Discovery AI - Quick Dataset Generator"
echo "=================================================="

cd "$(dirname "$0")"

# Create a minimal dataset generator that works with just Python
python3 << 'PYTHON_SCRIPT'
import json
import random
import os

# Ensure directories exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/images', exist_ok=True)

print("\n📊 Generating CSV data...")

# Generate CSV data (simple version)
csv_lines = ['drug_id,molecular_weight,logP,h_bond_donors,h_bond_acceptors,rotatable_bonds,bioavailability,half_life,clearance,hepatotoxicity_score,cardiotoxicity_score,solubility,melting_point,pKa,efficacy_score,safety_score,drug_category,development_stage,target']

categories = ['Antibiotic', 'Antiviral', 'Anticancer', 'Cardiovascular']
stages = ['Discovery', 'Preclinical', 'Phase I', 'Phase II', 'Phase III', 'Approved']

for i in range(10000):
    drug_id = f"DRUG_{str(i).zfill(6)}"
    molecular_weight = random.uniform(200, 600)
    logP = random.uniform(0, 5)
    h_bond_donors = random.randint(0, 8)
    h_bond_acceptors = random.randint(0, 12)
    rotatable_bonds = random.randint(0, 15)
    bioavailability = random.uniform(20, 95)
    half_life = random.uniform(1, 30)
    clearance = random.uniform(0.5, 20)
    hepatotoxicity_score = random.uniform(0, 10)
    cardiotoxicity_score = random.uniform(0, 10)
    solubility = random.uniform(-3, 6)
    melting_point = random.uniform(100, 300)
    pKa = random.uniform(3, 12)
    
    # Simple effectiveness calculation
    efficacy_score = (bioavailability/10 - hepatotoxicity_score/2 - cardiotoxicity_score/2 + random.uniform(-2, 2))
    efficacy_score = max(1, min(10, efficacy_score))
    
    safety_score = (10 - (hepatotoxicity_score + cardiotoxicity_score) / 2)
    safety_score = max(1, min(10, safety_score))
    
    target = 1 if (efficacy_score > 6 and safety_score > 6) else 0
    
    category = random.choice(categories)
    stage = random.choice(stages)
    
    line = f"{drug_id},{molecular_weight:.2f},{logP:.2f},{h_bond_donors},{h_bond_acceptors},{rotatable_bonds},{bioavailability:.2f},{half_life:.2f},{clearance:.2f},{hepatotoxicity_score:.2f},{cardiotoxicity_score:.2f},{solubility:.2f},{melting_point:.2f},{pKa:.2f},{efficacy_score:.2f},{safety_score:.2f},{category},{stage},{target}"
    csv_lines.append(line)

with open('data/raw/drug_data.csv', 'w') as f:
    f.write('\n'.join(csv_lines))

print(f"✓ Generated 10,000 drug records in data/raw/drug_data.csv")

# Generate JSON data
print("\n📄 Generating JSON data...")

json_data = []
for i in range(10000):
    drug_id = f"DRUG_{str(i).zfill(6)}"
    record = {
        'drug_id': drug_id,
        'protein_targets': random.sample(['EGFR', 'VEGFR', 'BCR-ABL', 'HER2'], random.randint(1, 3)),
        'mechanism_of_action': random.choice(['Competitive inhibition', 'Enzyme inhibitor', 'Receptor antagonist']),
        'side_effects': random.sample(['Nausea', 'Headache', 'Fatigue', 'Dizziness'], random.randint(0, 3)),
        'trial_patients': random.randint(50, 2000),
        'response_rate': round(random.uniform(0.3, 0.9), 3)
    }
    json_data.append(record)

with open('data/raw/drug_interactions.json', 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"✓ Generated 10,000 interaction records in data/raw/drug_interactions.json")

print("\n" + "="*50)
print("✅ Dataset generation complete!")
print("="*50)
print("\nGenerated files:")
print("  - data/raw/drug_data.csv (10,000 records)")
print("  - data/raw/drug_interactions.json (10,000 records)")
print("\nNote: Image generation requires additional packages.")
print("      The CSV and JSON data are sufficient for training.")

PYTHON_SCRIPT

echo ""
echo "Next steps:"
echo "  1. Install dependencies: pip install -r requirements.txt"
echo "  2. Open Jupyter notebook: jupyter notebook"
echo "  3. Run notebooks/train_model.ipynb"
echo ""
