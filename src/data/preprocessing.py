"""
Data Preprocessing for Drug Discovery
Handles missing values, feature engineering, and data cleaning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Dict, List, Tuple, Optional


class DataPreprocessor:
    """Comprehensive data preprocessing pipeline"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = []
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mixed') -> pd.DataFrame:
        """
        Handle missing values with multiple strategies
        
        Strategies:
        - 'mean': Fill numeric with mean
        - 'median': Fill numeric with median
        - 'mode': Fill categorical with mode
        - 'knn': Use KNN imputation
        - 'mixed': Use different strategies for different column types (recommended)
        """
        print("\n=== Handling Missing Values ===")
        print(f"Missing values before:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")
        
        df_copy = df.copy()
        
        if strategy == 'mixed':
            # Identify column types
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            categorical_cols = df_copy.select_dtypes(include=['object']).columns
            
            # Handle numeric columns with median
            if len(numeric_cols) > 0:
                imputer_numeric = SimpleImputer(strategy='median')
                df_copy[numeric_cols] = imputer_numeric.fit_transform(df_copy[numeric_cols])
                self.imputers['numeric'] = imputer_numeric
            
            # Handle categorical columns with most frequent
            if len(categorical_cols) > 0:
                imputer_categorical = SimpleImputer(strategy='most_frequent')
                df_copy[categorical_cols] = imputer_categorical.fit_transform(df_copy[categorical_cols])
                self.imputers['categorical'] = imputer_categorical
                
        elif strategy == 'knn':
            # KNN imputation (only for numeric columns)
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                imputer = KNNImputer(n_neighbors=5)
                df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])
                self.imputers['knn'] = imputer
        
        else:
            # Simple strategy for all columns
            imputer = SimpleImputer(strategy=strategy if strategy in ['mean', 'median', 'most_frequent'] else 'median')
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])
                self.imputers['simple'] = imputer
        
        print(f"Missing values after:\n{df_copy.isnull().sum()[df_copy.isnull().sum() > 0]}")
        print("✓ Missing values handled\n")
        return df_copy
    
    def create_interaction_features(self, df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features between pairs of features"""
        print("\n=== Creating Interaction Features ===")
        df_copy = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df_copy.columns and feat2 in df_copy.columns:
                # Check if numeric
                if pd.api.types.is_numeric_dtype(df_copy[feat1]) and pd.api.types.is_numeric_dtype(df_copy[feat2]):
                    # Multiplication
                    df_copy[f'{feat1}_x_{feat2}'] = df_copy[feat1] * df_copy[feat2]
                    
                    # Division (avoid division by zero)
                    df_copy[f'{feat1}_div_{feat2}'] = df_copy[feat1] / (df_copy[feat2] + 1e-8)
                    
                    # Addition
                    df_copy[f'{feat1}_plus_{feat2}'] = df_copy[feat1] + df_copy[feat2]
                    
                    print(f"✓ Created interactions for {feat1} and {feat2}")
        
        return df_copy
    
    def create_polynomial_features(self, df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for specified columns"""
        print("\n=== Creating Polynomial Features ===")
        df_copy = df.copy()
        
        for col in columns:
            if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                for d in range(2, degree + 1):
                    df_copy[f'{col}_pow{d}'] = df_copy[col] ** d
                    print(f"✓ Created {col}^{d}")
        
        return df_copy
    
    def create_binning_features(self, df: pd.DataFrame, columns: List[str], n_bins: int = 5) -> pd.DataFrame:
        """Create binned versions of continuous features"""
        print("\n=== Creating Binned Features ===")
        df_copy = df.copy()
        
        for col in columns:
            if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[f'{col}_binned'] = pd.qcut(df_copy[col], q=n_bins, labels=False, duplicates='drop')
                print(f"✓ Binned {col} into {n_bins} categories")
        
        return df_copy
    
    def create_statistical_features(self, df: pd.DataFrame, group_col: str, agg_cols: List[str]) -> pd.DataFrame:
        """Create statistical aggregation features"""
        print("\n=== Creating Statistical Features ===")
        df_copy = df.copy()
        
        if group_col not in df_copy.columns:
            return df_copy
        
        for col in agg_cols:
            if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                # Group statistics
                group_stats = df_copy.groupby(group_col)[col].agg(['mean', 'std', 'min', 'max'])
                group_stats.columns = [f'{col}_group_{stat}' for stat in ['mean', 'std', 'min', 'max']]
                
                df_copy = df_copy.merge(group_stats, left_on=group_col, right_index=True, how='left')
                print(f"✓ Created group statistics for {col}")
        
        return df_copy
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str] = None, method: str = 'label') -> pd.DataFrame:
        """Encode categorical variables"""
        print("\n=== Encoding Categorical Variables ===")
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.select_dtypes(include=['object']).columns
        
        for col in columns:
            if col in df_copy.columns:
                if method == 'label':
                    le = LabelEncoder()
                    df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                    self.encoders[col] = le
                    print(f"✓ Label encoded {col}")
                elif method == 'onehot':
                    dummies = pd.get_dummies(df_copy[col], prefix=col)
                    df_copy = pd.concat([df_copy, dummies], axis=1)
                    df_copy.drop(col, axis=1, inplace=True)
                    print(f"✓ One-hot encoded {col}")
        
        return df_copy
    
    def scale_features(self, df: pd.DataFrame, columns: List[str] = None, method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features"""
        print("\n=== Scaling Features ===")
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns
        
        columns = [col for col in columns if col in df_copy.columns]
        
        if len(columns) == 0:
            return df_copy
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
        self.scalers[method] = scaler
        print(f"✓ Scaled {len(columns)} features using {method} scaling")
        
        return df_copy
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers from specified columns"""
        print("\n=== Removing Outliers ===")
        df_copy = df.copy()
        initial_rows = len(df_copy)
        
        for col in columns:
            if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                if method == 'iqr':
                    Q1 = df_copy[col].quantile(0.25)
                    Q3 = df_copy[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    df_copy = df_copy[(df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)]
                
                elif method == 'zscore':
                    z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                    df_copy = df_copy[z_scores < threshold]
        
        removed_rows = initial_rows - len(df_copy)
        print(f"✓ Removed {removed_rows} outlier rows ({removed_rows/initial_rows*100:.2f}%)")
        
        return df_copy
    
    def feature_engineering_pipeline(self, df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        print("\n" + "="*50)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*50)
        
        df_processed = df.copy()
        
        # 1. Handle missing values
        df_processed = self.handle_missing_values(df_processed, strategy='mixed')
        
        # 2. Identify feature types
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from feature lists
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        # 3. Create interaction features (for first few numeric features)
        if len(numeric_cols) >= 2:
            feature_pairs = [(numeric_cols[i], numeric_cols[i+1]) for i in range(min(3, len(numeric_cols)-1))]
            df_processed = self.create_interaction_features(df_processed, feature_pairs)
        
        # 4. Create polynomial features (for important numeric features)
        if len(numeric_cols) >= 1:
            poly_cols = numeric_cols[:min(3, len(numeric_cols))]
            df_processed = self.create_polynomial_features(df_processed, poly_cols, degree=2)
        
        # 5. Encode categorical variables
        if len(categorical_cols) > 0:
            df_processed = self.encode_categorical(df_processed, categorical_cols, method='label')
        
        # 6. Remove outliers
        if len(numeric_cols) > 0:
            outlier_cols = numeric_cols[:min(5, len(numeric_cols))]
            df_processed = self.remove_outliers(df_processed, outlier_cols, method='iqr', threshold=3.0)
        
        # 7. Scale features (do this last, before model training)
        # We'll skip scaling here as it should be done after train-test split
        
        self.feature_names = df_processed.columns.tolist()
        
        print("\n" + "="*50)
        print(f"✓ Feature Engineering Complete!")
        print(f"Original features: {df.shape[1]}")
        print(f"Engineered features: {df_processed.shape[1]}")
        print(f"Original samples: {df.shape[0]}")
        print(f"Final samples: {df_processed.shape[0]}")
        print("="*50 + "\n")
        
        return df_processed
