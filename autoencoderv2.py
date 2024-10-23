import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, Model 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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
patient_dirs_healhty='EEG_data\EEG_data\Healthy'
 
eeg_data_healthy = load_all_eeg_data(patient_dirs_healhty) 
eeg_data_AD= load_all_eeg_data(patient_dirs) 
allEEG=np.vstack((eeg_data_healthy, eeg_data_AD))

autoencoder.compile(optimizer='adam', loss='mse') 
 

autoencoder.fit(allEEG, allEEG, epochs=50) 
encoder= autoencoder.encoder 

latent_features_healthy = encoder.predict(eeg_data_healthy) 
latent_features_AD = encoder.predict(eeg_data_AD) 
labels_healthy = np.zeros(latent_features_healthy.shape[0])  
labels_unhealthy = np.ones(latent_features_AD.shape[0]) 
X = np.vstack((latent_features_healthy, latent_features_AD))   
y = np.concatenate((labels_healthy, labels_unhealthy))  
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], label='Healthy', alpha=0.7)
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], label='AD', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Latent Features')
plt.legend()
plt.show()

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], label='Healthy', alpha=0.7)
plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], label='AD', alpha=0.7)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE of Latent Features')
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")