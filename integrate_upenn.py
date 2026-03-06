"""
Integration script: UPenn data -> VEP pipeline
"""

import sys
sys.path.append('./vep')

from upenn_loader import UPennLoader
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Initialize loader
    upenn = UPennLoader(data_root='./data/upenn')
    
    # List available patients
    patients = upenn.list_patients()
    print(f"Found {len(patients)} patients")
    print("Available patients:", patients[:5])  # Show first 5
    
    if not patients:
        print("No patient data found!")
        print("   Download data to: ./data/upenn/")
        return
    
    # Load first patient
    patient_id = patients[0]
    print(f"\nLoading patient: {patient_id}")
    
    try:
        raw, events, electrodes = upenn.load_patient_ieeg(patient_id)
        
        print(f"SUCCESS - iEEG data loaded:")
        print(f"  - Channels: {len(raw.ch_names)}")
        print(f"  - Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  - Duration: {raw.times[-1]:.1f} seconds")
        print(f"  - Events: {len(events)}")
        print(f"  - Electrodes: {len(electrodes)}")
        
        # Get electrode positions
        coords, labels, regions = upenn.get_electrode_positions(electrodes)
        print(f"\nElectrode positions:")
        for i in range(min(5, len(labels))):
            print(f"  {labels[i]}: ({coords[i,0]:.1f}, {coords[i,1]:.1f}, {coords[i,2]:.1f}) - {regions[i]}")
        
        # Find seizures
        seizures = upenn.find_seizures(events)
        if len(seizures) > 0:
            print(f"\nFOUND {len(seizures)} seizure events!")
            print(seizures[['onset', 'duration', 'trial_type']].head())
        
        # Plot raw data sample (first 10 seconds, 10 channels)
        fig = raw.plot(duration=10, n_channels=10, scalings='auto', show=False)
        plt.savefig('upenn_sample.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\nSaved plot: upenn_sample.png")
        
    except Exception as e:
        print(f"ERROR loading patient: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()