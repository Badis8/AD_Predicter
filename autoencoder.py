import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, Model
latent_dim = 64

class EEGAutoencoder(Model):
    def __init__(self, latent_dim):
        super(EEGAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),  
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(22 * 1025, activation='sigmoid'),   
            layers.Reshape((22, 1025))   
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

 
autoencoder = EEGAutoencoder(latent_dim)

def load_patient_data(patient_dir):
    eeg_data = []
    for electrode_file in sorted(os.listdir(patient_dir)):
        file_path = os.path.join(patient_dir, electrode_file)
        electrode_data = np.loadtxt(file_path)
        
 
        electrode_data = (electrode_data - np.min(electrode_data)) / (np.max(electrode_data) - np.min(electrode_data))
        
        if electrode_data.shape[0] == 1025:
            eeg_data.append(electrode_data)

    eeg_array = np.array(eeg_data)
    
    if eeg_array.shape[0] == 22:
        return eeg_array
    elif eeg_array.shape[0] < 22:
        padding = np.zeros((22 - eeg_array.shape[0], 1025))
        eeg_array = np.vstack([eeg_array, padding])
        return eeg_array
    else:
        return eeg_array[:22]

def load_all_eeg_data(patient_dirs):
    all_eeg_data = []   

 
    patient_dir_closed = os.path.join(patient_dirs, "Eyes_closed")
    patient_dir_open = os.path.join(patient_dirs, "Eyes_open")

 
    for patient_dir in sorted(os.listdir(patient_dir_closed)):
        full_path = os.path.join(patient_dir_closed, patient_dir)
        if os.path.isdir(full_path):   
            eeg_flat = load_patient_data(full_path)
            all_eeg_data.append(eeg_flat)

  
    for patient_dir in sorted(os.listdir(patient_dir_open)):
        full_path = os.path.join(patient_dir_open, patient_dir)
        if os.path.isdir(full_path):  
            eeg_flat = load_patient_data(full_path)
            all_eeg_data.append(eeg_flat)

    
    all_eeg_data = np.stack(all_eeg_data)   
    return all_eeg_data

patient_dirs = 'EEG_data\EEG_data\AD'

 
eeg_data = load_all_eeg_data(patient_dirs)
autoencoder.compile(optimizer='adam', loss='mse') 
autoencoder.fit(eeg_data, eeg_data, epochs=50)
autoencoder.save('eeg_autoencoder.keras')
 
print(f"EEG data shape: {eeg_data.shape}")