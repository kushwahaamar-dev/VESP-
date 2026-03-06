"""
UPenn/OpenNeuro iEEG Data Loader
Loads patient data from OpenNeuro ds004789 dataset
"""

import os
import pandas as pd
import mne
import numpy as np

class UPennLoader:
    def __init__(self, data_root="./data/upenn"):
        self.data_root = data_root
        
    def list_patients(self):
        """List available patient IDs"""
        patients = [d for d in os.listdir(self.data_root) 
                   if d.startswith('sub-')]
        return sorted(patients)
    
    def load_patient_ieeg(self, patient_id, session=None, run=None):
        """
        Load iEEG recording for a patient
        """
        patient_dir = os.path.join(self.data_root, patient_id)
        sessions = [d for d in os.listdir(patient_dir) if d.startswith('ses-')]
        
        if not sessions:
            raise ValueError(f"No sessions found for {patient_id}")
            
        if session is None:
            session = sessions[0]
        
        ieeg_dir = os.path.join(patient_dir, session, 'ieeg')
        
        edf_files = [f for f in os.listdir(ieeg_dir) if f.endswith('_ieeg.edf')]
        if not edf_files:
            raise ValueError(f"No iEEG files found in {ieeg_dir}")
        
        if run is not None:
            edf_file = [f for f in edf_files if f'run-{run:02d}' in f][0]
        else:
            edf_file = edf_files[0]
        
        edf_path = os.path.join(ieeg_dir, edf_file)
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        
        base_name = edf_file.replace('_ieeg.edf', '')
        
        events_file = base_name + '_events.tsv'
        events_path = os.path.join(ieeg_dir, events_file)
        
        events_df = None
        if not os.path.exists(events_path):
            events_files = [f for f in os.listdir(ieeg_dir) if 'events.tsv' in f]
            if events_files:
                events_path = os.path.join(ieeg_dir, events_files[0])
            else:
                events_df = pd.DataFrame(columns=['onset', 'duration', 'trial_type'])
        
        if events_df is None and os.path.exists(events_path):
            events_df = pd.read_csv(events_path, sep='\t')
        
        electrodes_file = base_name + '_electrodes.tsv'
        electrodes_path = os.path.join(ieeg_dir, electrodes_file)
        
        if not os.path.exists(electrodes_path):
            electrode_files = [f for f in os.listdir(ieeg_dir) if 'electrodes.tsv' in f]
            if electrode_files:
                electrodes_path = os.path.join(ieeg_dir, electrode_files[0])
            else:
                electrodes_df = pd.DataFrame({
                    'name': raw.ch_names,
                    'x': [0]*len(raw.ch_names),
                    'y': [0]*len(raw.ch_names),
                    'z': [0]*len(raw.ch_names)
                })
                return raw, events_df, electrodes_df
        
        electrodes_df = pd.read_csv(electrodes_path, sep='\t')
        return raw, events_df, electrodes_df
    
    def get_electrode_positions(self, electrodes_df):
        """Extract 3D coordinates from electrodes dataframe"""
        coords = electrodes_df[['x', 'y', 'z']].values
        labels = electrodes_df['name'].values
        
        if 'region' in electrodes_df.columns:
            regions = electrodes_df['region'].values
        else:
            regions = np.array(['']*len(labels))
        
        return coords, labels, regions
    
    def find_seizures(self, events_df):
        """Find seizure events in the event data"""
        seizure_events = events_df[
            events_df['trial_type'].str.contains('seizure', case=False, na=False)
        ]
        return seizure_events