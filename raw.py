import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class EEGDataClassifier:
    def __init__(self, root_path):
        self.root_path = root_path
        self.df_raw = pd.DataFrame()
        self.df_stats=pd.DataFrame()

    def load_stat_data_from_folder(self, folder_path, label):
        data_list = []

        for condition in ['Eyes_closed', 'Eyes_open']:
            condition_path = os.path.join(folder_path, condition)
            eyes_state = 0 if condition == 'Eyes_closed' else 1

            for patient_folder in os.listdir(condition_path):
                patient_path = os.path.join(condition_path, patient_folder)

                if not os.path.isdir(patient_path):
                    continue

                patient_features = {}

                for csv_file in os.listdir(patient_path):
                    if csv_file.endswith('.csv'):
                        csv_file_path = os.path.join(patient_path, csv_file)
                        df = pd.read_csv(csv_file_path, header=None)
                        data = df[0].tolist()

               
                        feature_name = os.path.splitext(csv_file)[0]
                        patient_features[f'{feature_name}_mean'] = np.mean(data)
                        patient_features[f'{feature_name}_var'] = np.var(data)
                        patient_features[f'{feature_name}_min'] = np.min(data)
                        patient_features[f'{feature_name}_max'] = np.max(data)

                if not patient_features:
                    print(f"No features found for patient: {patient_folder} in condition: {condition}")
                    continue

      
                row_data = {key: patient_features[key] for key in patient_features}
              
                row_data['Label'] = label
                data_list.append(row_data)

        return pd.DataFrame(data_list)


    def load_raw_data_from_folder(self, folder_path, label):
        data_list = []
        for condition in ['Eyes_closed', 'Eyes_open']:
            condition_path = os.path.join(folder_path, condition)
            eyes_state = 0 if condition == 'Eyes_closed' else 1

            for patient_folder in os.listdir(condition_path):
                patient_path = os.path.join(condition_path, patient_folder)

                if not os.path.isdir(patient_path):
                    continue

                patient_features = {}

                for csv_file in os.listdir(patient_path):
                    if csv_file.endswith('.csv'):
                        csv_file_path = os.path.join(patient_path, csv_file)
                        df = pd.read_csv(csv_file_path, header=None)

                        
                        for index, value in enumerate(df[0].tolist()):
                            feature_name = f'{os.path.splitext(csv_file)[0]}_{index + 1}'   
                            patient_features[feature_name] = value   
                if not patient_features:
                    print(f"No features found for patient: {patient_folder} in condition: {condition}")
                    continue

                row_data = {key: patient_features[key] for key in patient_features}
                row_data['Label'] = label
                data_list.append(row_data)

        return pd.DataFrame(data_list)

    def load_data(self):
        healthy_data = self.load_raw_data_from_folder(os.path.join(self.root_path, 'Healthy'), label=0)
        ad_data = self.load_raw_data_from_folder(os.path.join(self.root_path, 'AD'), label=1) 
    

        stat_healthy_data = self.load_stat_data_from_folder(os.path.join(self.root_path, 'Healthy'), label=0)
        stat_ad_data = self.load_stat_data_from_folder(os.path.join(self.root_path, 'AD'), label=1) 

        self.df_stats = pd.concat([stat_healthy_data, stat_ad_data], ignore_index=True)
        self.df_raw = pd.concat([healthy_data, ad_data], ignore_index=True) 



        if not self.df_raw.empty:
            num_features = self.df_raw.shape[1] - 2
            self.df_raw.columns = [f'Feature_{i}' for i in range(num_features)] + ['Eyes_state', 'Label']

    def train_and_evaluate(self):
        X = self.df_raw.iloc[:, :-1].values
        y = self.df_raw.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = DecisionTreeClassifier(random_state=42)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred) 
        print(f"Accuracy of the Decision Tree raw: {accuracy:.2f}")  


        stat_information = self.df_stats.drop(columns=['Label'])
        stat_labels = self.df_stats['Label']

        stat_information_train, stat_information_test, stat_labels_train, stat_labels_test = train_test_split(stat_information, stat_labels, test_size=0.2, random_state=42)
        clf = DecisionTreeClassifier(random_state=42)

        clf.fit(stat_information_train, stat_labels_train)
        y_pred_stat = clf.predict(stat_information_test)

        accuracy_stat = accuracy_score(stat_labels_test, y_pred_stat) 
        print(f"Accuracy of the Decision Tree raw: {accuracy_stat:.2f}")  



if __name__ == "__main__":
    root_path = 'EEG_data\\EEG_data'
    classifier = EEGDataClassifier(root_path)
    classifier.load_data()

  
    print("Statistical Data (df_stats):")
    print(classifier.df_raw.head())  

    classifier.train_and_evaluate()