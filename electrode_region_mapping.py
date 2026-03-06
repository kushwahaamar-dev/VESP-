"""
Map electrodes to brain regions
"""
import sys
sys.path.append('./vep')

from upenn_loader import UPennLoader
import pandas as pd

def map_electrodes_to_regions(patient_id):
    """Map each electrode to its brain region"""
    upenn = UPennLoader(data_root='./data/upenn')
    
    print(f"\nMapping {patient_id}...")
    raw, events, electrodes = upenn.load_patient_ieeg(patient_id)
    
    # Create mapping table
    mapping = []
    for idx, row in electrodes.iterrows():
        region = row.get('ind.region', 'unknown')
        if pd.isna(region) or region == 'n/a':
            region = 'unknown'
        
        mapping.append({
            'Electrode': row['name'],
            'X': f"{row['x']:.1f}",
            'Y': f"{row['y']:.1f}",
            'Z': f"{row['z']:.1f}",
            'Brain_Region': region
        })
    
    df = pd.DataFrame(mapping)
    
    # Save to CSV
    filename = f'{patient_id}_electrode_mapping.csv'
    df.to_csv(filename, index=False)
    print(f"  Saved: {filename}")
    
    # Print sample
    print(f"\n  Sample mapping:")
    print(df.head(10).to_string(index=False))
    
    # Region summary
    region_counts = df['Brain_Region'].value_counts()
    print(f"\n  Regions covered ({len(region_counts)} total):")
    for region, count in region_counts.head(5).items():
        print(f"    {region}: {count} electrodes")
    
    return len(df), len(region_counts)

patients = ['sub-R1001P', 'sub-R1002P', 'sub-R1003P']

print("="*60)
print("ELECTRODE-TO-REGION MAPPING")
print("="*60)

for patient in patients:
    n_elec, n_regions = map_electrodes_to_regions(patient)
    print(f"\n  Total: {n_elec} electrodes across {n_regions} regions")
    print("-"*60)

print("\n" + "="*60)
print("MAPPING COMPLETE!")
print("="*60)
