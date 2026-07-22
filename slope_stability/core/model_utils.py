import numpy as np
import pandas as pd
import joblib
import os


class EnsemblePredictor:
    REQUIRED_FILES = {
        'scaler': 'scaler.pkl',
        'weights': 'weights.pkl',
    }
    
    MODEL_FILES = {
        'xgb': 'model_xgb.json',
        'cat': 'model_cat.cbm',
        'lgb': 'model_lgb.pkl',
        'rf': 'model_rf.pkl',
        'et': 'model_et.pkl',
        'gb': 'model_gb.pkl',
    }

    def __init__(self, model_dir):
        self.model_dir = os.path.abspath(model_dir)
        self.scaler = None
        self.weights = None
        self.models = {}
        self.load_errors = []

    def check_model_directory(self):
        if not os.path.exists(self.model_dir):
            self.load_errors.append(f"Model directory does not exist: {self.model_dir}")
            return False
        
        if not os.path.isdir(self.model_dir):
            self.load_errors.append(f"Model path is not a directory: {self.model_dir}")
            return False
        
        return True

    def check_required_files(self):
        missing_files = []
        for key, filename in self.REQUIRED_FILES.items():
            filepath = os.path.join(self.model_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)
        
        if missing_files:
            self.load_errors.append(f"Missing required files: {', '.join(missing_files)}")
            return False
        return True

    def load(self):
        self.load_errors = []
        
        if not self.check_model_directory():
            raise FileNotFoundError("\n".join(self.load_errors))
        
        if not self.check_required_files():
            raise FileNotFoundError("\n".join(self.load_errors))
        
        try:
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        except Exception as e:
            raise RuntimeError(f"Failed to load scaler.pkl: {str(e)}")
        
        try:
            self.weights = joblib.load(os.path.join(self.model_dir, 'weights.pkl'))
        except Exception as e:
            raise RuntimeError(f"Failed to load weights.pkl: {str(e)}")
        
        for name, filename in self.MODEL_FILES.items():
            filepath = os.path.join(self.model_dir, filename)
            if os.path.exists(filepath):
                try:
                    if name == 'xgb':
                        import xgboost as xgb
                        self.models['xgb'] = xgb.XGBClassifier()
                        self.models['xgb'].load_model(filepath)
                    elif name == 'cat':
                        from catboost import CatBoostClassifier
                        self.models['cat'] = CatBoostClassifier()
                        self.models['cat'].load_model(filepath)
                    else:
                        self.models[name] = joblib.load(filepath)
                except Exception as e:
                    self.load_errors.append(f"Failed to load {filename}: {str(e)}")
        
        if not self.models:
            raise RuntimeError(f"No models were loaded. Check if model files exist in: {self.model_dir}")
        
        return self

    def predict_proba(self, X):
        if self.scaler is None:
            raise RuntimeError("Scaler not loaded. Call load() first.")
        
        X_scaled = self.scaler.transform(X)
        
        weighted_proba = np.zeros(len(X), dtype=float)
        
        for name, model in self.models.items():
            weight = self.weights.get(name, 0) if self.weights else 0
            if weight > 0:
                proba = model.predict_proba(X_scaled)[:, 1]
                weighted_proba += proba * weight
        
        return weighted_proba

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)