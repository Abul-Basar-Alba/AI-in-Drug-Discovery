"""
Data Loader for Drug Discovery Project
Handles CSV, JSON, and Image data
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional


class DrugDataLoader:
    """Load and combine multiple data sources for drug discovery"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        
    def load_csv_data(self, filename: str) -> pd.DataFrame:
        """Load CSV data with drug properties"""
        filepath = self.data_dir / filename
        print(f"Loading CSV data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records from CSV")
        return df
    
    def load_json_data(self, filename: str) -> pd.DataFrame:
        """Load JSON data with drug interactions and metadata"""
        filepath = self.data_dir / filename
        print(f"Loading JSON data from {filepath}...")
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert JSON to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        
        print(f"Loaded {len(df)} records from JSON")
        return df
    
    def load_image_features(self, image_dir: str = "images") -> pd.DataFrame:
        """Extract features from molecular structure images"""
        img_path = self.data_dir.parent / image_dir
        print(f"Loading images from {img_path}...")
        
        image_features = []
        image_files = list(img_path.glob("*.png")) + list(img_path.glob("*.jpg"))
        
        for img_file in image_files:
            try:
                # Load image
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                # Extract basic features
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Color histogram features
                hist_b = cv2.calcHist([img], [0], None, [32], [0, 256]).flatten()
                hist_g = cv2.calcHist([img], [1], None, [32], [0, 256]).flatten()
                hist_r = cv2.calcHist([img], [2], None, [32], [0, 256]).flatten()
                
                # Texture features (using standard deviation of pixel intensities)
                texture = np.std(img_gray)
                
                # Mean brightness
                brightness = np.mean(img_gray)
                
                # Edge density
                edges = cv2.Canny(img_gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                
                features = {
                    'drug_id': img_file.stem,
                    'brightness': brightness,
                    'texture_std': texture,
                    'edge_density': edge_density,
                }
                
                # Add histogram features
                for i, val in enumerate(hist_r[:8]):  # Use only first 8 bins
                    features[f'hist_r_{i}'] = val
                for i, val in enumerate(hist_g[:8]):
                    features[f'hist_g_{i}'] = val
                for i, val in enumerate(hist_b[:8]):
                    features[f'hist_b_{i}'] = val
                
                image_features.append(features)
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        df = pd.DataFrame(image_features)
        print(f"Extracted features from {len(df)} images")
        return df
    
    def merge_all_data(self, csv_file: str, json_file: str, 
                       use_images: bool = True) -> pd.DataFrame:
        """Merge all data sources into a single DataFrame"""
        print("\n=== Loading All Data Sources ===")
        
        # Load CSV (primary data)
        df_csv = self.load_csv_data(csv_file)
        
        # Load JSON
        df_json = self.load_json_data(json_file)
        
        # Merge CSV and JSON
        if 'drug_id' in df_csv.columns and 'drug_id' in df_json.columns:
            df_merged = pd.merge(df_csv, df_json, on='drug_id', how='left', suffixes=('', '_json'))
        else:
            df_merged = df_csv
        
        # Load and merge image features if available
        if use_images:
            try:
                df_images = self.load_image_features()
                if not df_images.empty and 'drug_id' in df_merged.columns:
                    df_merged = pd.merge(df_merged, df_images, on='drug_id', how='left')
            except Exception as e:
                print(f"Warning: Could not load image features: {e}")
        
        print(f"\n=== Final merged dataset: {df_merged.shape[0]} rows, {df_merged.shape[1]} columns ===\n")
        return df_merged


class TextDataProcessor:
    """Process text data for NLP tasks in drug discovery"""
    
    @staticmethod
    def extract_text_features(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """Extract NLP features from text columns"""
        print("Extracting text features...")
        
        for col in text_columns:
            if col in df.columns:
                # Text length
                df[f'{col}_length'] = df[col].fillna('').astype(str).apply(len)
                
                # Word count
                df[f'{col}_word_count'] = df[col].fillna('').astype(str).apply(lambda x: len(x.split()))
                
                # Character diversity (unique chars / total chars)
                df[f'{col}_char_diversity'] = df[col].fillna('').astype(str).apply(
                    lambda x: len(set(x)) / len(x) if len(x) > 0 else 0
                )
        
        print(f"Added text features for {len(text_columns)} columns")
        return df
