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
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE
from datetime import datetime


class DataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.selected_features = []

    # ------------------ LOAD DATA ------------------
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
        print(f"Loaded dataset: {df.shape}")
        return df

    # ------------------ DATA INFO ------------------
    def check_data_info(self, df):
        print("\n--- DATA INFO ---")
        print(df.info())
        print("\nSUMMARY STATISTICS\n", df.describe())
        print("\nMISSING VALUE (%)\n", (df.isnull().sum() / len(df) * 100).round(2))
        print("\nUNIQUE VALUES\n", df.nunique())

    # ------------------ PLOT MISSING VALUES ------------------
    def plot_missing(self, df):
        print("Visualizing missing values...")
        plt.figure(figsize=(12, 4))
        msno.matrix(df)
        plt.show()

        plt.figure(figsize=(10, 5))
        msno.heatmap(df)
        plt.show()

        plt.figure(figsize=(10, 5))
        msno.bar(df)
        plt.show()

    # ------------------ PLOT CLASS BALANCE ------------------
    def plot_class_balance(self, df, target='Class'):
        if target not in df.columns:
            print(f"Target column '{target}' not found")
            return
        plt.figure(figsize=(6, 4))
        sns.countplot(x=target, data=df)
        plt.title("Class Distribution")
        plt.show()
        print(f"Class distribution:\n{df[target].value_counts()}")

    # ------------------ HANDLE MISSING VALUES ------------------
    def handle_missing_values(self, df, zero_cols=None, drop_threshold=0.3):
        df = df.copy()
        
        # Columns where 0 should be treated as missing
        if zero_cols is None:
            zero_cols = ['Glucose', 'Diastolic_BP', 'Skin_Fold', 'Serum_Insulin', 'BMI']

        # Replace 0 with NaN for known columns
        for col in zero_cols:
            if col in df.columns:
                zeros = (df[col] == 0).sum()
                if zeros > 0:
                    print(f"Replaced {zeros} zeros in '{col}' → NaN")
                    df[col] = df[col].replace(0, np.nan)

        # Missing % by column
        missing_pct = df.isnull().mean()
        print("\nMISSING VALUE % PER COLUMN:\n", (missing_pct * 100).round(2))

        # Initial numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Fill or drop columns based on missing %
        for col in numeric_cols:
            if col not in df.columns:
                continue  # skip columns already dropped
            if missing_pct[col] == 0:
                continue
            elif missing_pct[col] < 0.05:
                df[col] = df[col].fillna(df[col].mean())
                print(f"'{col}': filled with MEAN ({missing_pct[col]*100:.1f}%)")
            elif missing_pct[col] < drop_threshold:
                df[col] = df[col].fillna(df[col].median())
                print(f"'{col}': filled with MEDIAN ({missing_pct[col]*100:.1f}%)")
            else:
                print(f"'{col}': dropped ({missing_pct[col]*100:.1f}%)")
                df = df.drop(columns=[col])

        # Recalculate numeric columns after drops
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Final safety check — use KNN Imputer if some missing remain
        if df[numeric_cols].isnull().sum().sum() > 0:
            print("Running KNN Imputer for remaining missing values...")
            imputer = KNNImputer(n_neighbors=3)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        return df

    # ------------------ REMOVE DUPLICATES ------------------
    def remove_duplicates(self, df):
        dup = df.duplicated().sum()
        if dup > 0:
            df = df.drop_duplicates()
            print(f"Removed {dup} duplicate rows")
        else:
            print("No duplicates found")
        return df

    # ------------------ OUTLIER CAPPING ------------------
    def detect_and_cap_outliers(self, df):
        df = df.copy()
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                df[col] = df[col].clip(lower, upper)
                print(f"Capped {outliers} outliers in '{col}'")
        return df

    # ------------------ FEATURE ENGINEERING ------------------
    def feature_engineering(self, df):
        df = df.copy()
        if 'BMI' in df.columns:
            df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100],
                                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        if 'Age' in df.columns:
            df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 100],
                                     labels=['Young', 'Middle-Aged', 'Senior'])
        if 'Glucose' in df.columns:
            df['Glucose_Level'] = pd.cut(df['Glucose'], bins=[0, 99, 125, 1000],
                                         labels=['Normal', 'Prediabetes', 'Diabetes'])
        print("Added derived columns: BMI_Category, Age_Group, Glucose_Level")
        return df

    # ------------------ ENCODING ------------------
    def encode_categorical(self, df):
        df = df.copy()
        if 'BMI_Category' in df.columns:
            order = ['Underweight', 'Normal', 'Overweight', 'Obese']
            df['BMI_Category_Encoded'] = pd.Categorical(df['BMI_Category'], categories=order, ordered=True).codes
        df = pd.get_dummies(df, columns=[c for c in ['Age_Group', 'Glucose_Level'] if c in df.columns], drop_first=True)
        print("Encoded categorical variables (label + one-hot)")
        return df

    # ------------------ SCALING ------------------
    def scale_features(self, df, cols=None):
        df = df.copy()
        if cols is None:
            cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64] and c != 'Class']
        self.scaler = MinMaxScaler()
        df[cols] = self.scaler.fit_transform(df[cols])
        print(f"Scaled {len(cols)} numeric features with MinMaxScaler")
        return df

    # ------------------ FEATURE SELECTION ------------------
    def select_features(self, df, k=8):
        df = df.copy()
        X = df.drop('Class', axis=1)
        X = X.select_dtypes(include=[np.number])
        y = df['Class']
        selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
        X_new = selector.fit_transform(X, y)
        self.selected_features = X.columns[selector.get_support()].tolist()
        print(f"Selected top {len(self.selected_features)} features: {self.selected_features}")
        df_new = pd.DataFrame(X_new, columns=self.selected_features)
        df_new['Class'] = y.values
        return df_new

    # ------------------ PCA ------------------
    def apply_pca(self, df, n=6):
        X = df.drop('Class', axis=1)
        y = df['Class']
        self.pca = PCA(n_components=min(n, X.shape[1]))
        X_pca = self.pca.fit_transform(X)
        variance = self.pca.explained_variance_ratio_.cumsum()[-1]
        print(f"PCA: {X_pca.shape[1]} components explain {variance:.1%} variance")
        df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
        df_pca['Class'] = y.values
        return df_pca

    # ------------------ SMOTE ------------------
    def handle_imbalance(self, df):
        X = df.drop('Class', axis=1)
        y = df['Class']
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        print(f"Before SMOTE: {dict(y.value_counts())}")
        print(f"After SMOTE: {dict(pd.Series(y_res).value_counts())}")
        df_bal = pd.DataFrame(X_res, columns=X.columns)
        df_bal['Class'] = y_res
        return df_bal


    # ------------------ FULL PIPELINE ------------------
    def full_pipeline(self, input_path, output_dir="data"):
        print("🚀 STARTING FULL PIPELINE\n" + "="*60)
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

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(output_dir, f"Cleaned_Diabetes_{timestamp}.csv")
        df.to_csv(output_path, index=False)
        print(f"\n✅ FINAL DATA SAVED: {output_path}")
        print(f"Final shape: {df.shape}")
        return df
