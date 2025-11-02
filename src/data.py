# src/preprocessing.py
import pandas as pd
import numpy as np
import os
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.selected_features = []

    def load_data(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith((".xlsx", ".xls")):
            df = pd.read_excel(filepath)
        elif filepath.endswith(".parquet"):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError("Only .csv, .xlsx, .parquet supported")
        print(f"Loaded: {df.shape}")
        return df

    def check_data_info(self, df):
        print("\nDATA INFO")
        print(df.info())
        print("\nSUMMARY")
        print(df.describe())
        print("\nMISSING %")
        print((df.isnull().sum() / len(df) * 100).round(2))
        print("\nUNIQUE VALUES")
        print(df.nunique())

    def handle_missing_values(self, df, zero_cols=None):
        df = df.copy()
        if zero_cols is None:
            zero_cols = ['Glucose', 'Diastolic_BP', 'Skin_Fold', 'Serum_Insulin', 'BMI']
        for col in zero_cols:
            if col in df.columns:
                zeros = (df[col] == 0).sum()
                if zeros > 0:
                    print(f"Replaced {zeros} zeros in '{col}' → NaN")
                    df[col] = df[col].replace(0, np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        print("Imputed all numeric columns with median")
        return df

    def remove_duplicates(self, df):
        dup = df.duplicated().sum()
        if dup > 0:
            df = df.drop_duplicates()
            print(f"Removed {dup} duplicates")
        else:
            print("No duplicates")
        return df

    def detect_and_cap_outliers(self, df):
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                df[col] = df[col].clip(lower, upper)
                print(f"Capped {outliers} outliers in '{col}'")
        return df

    def feature_engineering(self, df):
        df = df.copy()
        df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], 
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], 
                                labels=['Young', 'Middle-Aged', 'Senior'])
        df['Glucose_Level'] = pd.cut(df['Glucose'], bins=[0, 99, 125, 1000], 
                                    labels=['Normal', 'Prediabetes', 'Diabetes'])
        print("Added: BMI_Category, Age_Group, Glucose_Level")
        return df

    def encode_categorical(self, df):
        df = df.copy()
        order = ['Underweight', 'Normal', 'Overweight', 'Obese']
        df['BMI_Category_Encoded'] = pd.Categorical(df['BMI_Category'], categories=order, ordered=True).codes
        df = pd.get_dummies(df, columns=['Age_Group', 'Glucose_Level'], drop_first=True)
        print("Encoded BMI (label), Age/Glucose (one-hot)")
        return df

    def scale_features(self, df, cols=None):
        df = df.copy()
        if cols is None:
            cols = ['Pregnant', 'Glucose', 'Diastolic_BP', 'Skin_Fold', 
                    'Serum_Insulin', 'BMI', 'Diabetes_Pedigree', 'Age']
        self.scaler = MinMaxScaler()
        df[cols] = self.scaler.fit_transform(df[cols])
        print(f"Scaled {len(cols)} features with MinMaxScaler")
        return df

    def select_features(self, df, k=8):
        df = df.copy()
        X = df.drop('Class', axis=1)

        # Keep numeric columns only
        X = X.select_dtypes(include=[np.number])
        y = df['Class']

        selector = SelectKBest(mutual_info_classif, k=k)
        X_new = selector.fit_transform(X, y)

        self.selected_features = X.columns[selector.get_support()].tolist()
        print(f"Selected top {k} features: {self.selected_features}")

        df_new = pd.DataFrame(X_new, columns=self.selected_features)
        df_new['Class'] = y.values
        return df_new


    def apply_pca(self, df, n=6):
        X = df.drop('Class', axis=1)
        y = df['Class']
        self.pca = PCA(n_components=n)
        X_pca = self.pca.fit_transform(X)
        variance = self.pca.explained_variance_ratio_.cumsum()[-1]
        print(f"PCA: {n} components → {variance:.1%} variance")
        df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n)])
        df_pca['Class'] = y.values
        return df_pca

    def handle_imbalance(self, df):
        X = df.drop('Class', axis=1)
        y = df['Class']
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        print(f"Before: {dict(y.value_counts())}")
        print(f"After SMOTE: {dict(pd.Series(y_res).value_counts())}")
        df_bal = pd.DataFrame(X_res, columns=X.columns)
        df_bal['Class'] = y_res
        return df_bal

    def full_pipeline(self, input_path, output_path="data/processed_diabetes.csv"):
        print("STARTING FULL PIPELINE\n" + "="*50)
        df = self.load_data(input_path)
        self.check_data_info(df)
        
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        df = self.detect_and_cap_outliers(df)
        df = self.feature_engineering(df)
        df = self.encode_categorical(df)
        df = self.scale_features(df)
        df = self.select_features(df)
        df = self.apply_pca(df)
        df = self.handle_imbalance(df)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nFINAL DATASET SAVED: {output_path}")
        print(f"Shape: {df.shape}")
        return df