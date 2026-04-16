import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

# New Powerful Models
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim

# Fairness Libraries
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate, false_negative_rate, demographic_parity_ratio
from fairlearn.preprocessing import CorrelationRemover
from scipy.stats import chi2_contingency

# ─── 1. Deep Learning Architecture (PyTorch) ────────────────────────
class DeepLearningMLP(nn.Module):
    """A standard Feed-Forward Neural Network for tabular data."""
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Prevent overfitting
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()      # Output probability between 0 and 1
        )

    def forward(self, x):
        return self.network(x)

class PyTorchSklearnWrapper(BaseEstimator, ClassifierMixin):
    """Wraps PyTorch in a Scikit-Learn interface so it plays nicely with Fairlearn."""
    def __init__(self, epochs=50, lr=0.01):
        self.epochs = epochs
        self.lr = lr
        self.model = None

    def fit(self, X, y):
        # Convert Pandas DataFrames to PyTorch Tensors
        X_tensor = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X)
        y_tensor = torch.FloatTensor(y.values if isinstance(y, pd.Series) else y).unsqueeze(1)

        self.model = DeepLearningMLP(input_dim=X_tensor.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Training Loop
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
        return self

    def predict(self, X):
        X_tensor = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            # Convert probabilities to binary predictions (0 or 1)
            return (outputs.numpy() > 0.5).astype(int).flatten()

# ─── 2. The Fairness Engine ────────────────────────────────────────
class FairnessAnalyzer:
    def __init__(self):
        self.df = None
        self._last_config = {}
        
        # 🏆 THE UPGRADED ML FACTORY
        self.models = {
            "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000),
            "Random Forest (Non-Linear)": RandomForestClassifier(n_estimators=100),
            "XGBoost (Gradient Boosting)": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            "PyTorch Neural Network (Deep Learning)": PyTorchSklearnWrapper(epochs=100, lr=0.005)
        }

    def load_data(self, df: pd.DataFrame):
        self.df = df

    def evaluate_bias(self, target_column: str, protected_attribute: str, favorable_class: str, privileged_group: str):
        self._last_config = { "target_column": target_column, "protected_attribute": protected_attribute, "favorable_class": favorable_class, "privileged_group": privileged_group }
        
        y_true = (self.df[target_column].astype(str) == favorable_class).astype(int)
        
        # Standardize data for Neural Networks (Crucial step for Deep Learning stability)
        X_raw = pd.get_dummies(self.df.drop(columns=[target_column])).fillna(0)
        # Simple Min-Max scaling for the PyTorch/XGBoost models to digest easily
        X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min() + 1e-8)
        
        model_results = {}
        selection_rates = {}
        
        for name, model in self.models.items():
            # The interface remains completely uniform regardless of the backend engine
            model.fit(X, y_true)
            y_pred = model.predict(X)
            acc = accuracy_score(y_true, y_pred)
            
            mf = MetricFrame(metrics={"selection_rate": selection_rate}, y_true=y_true, y_pred=y_pred, sensitive_features=self.df[protected_attribute])
            sr_group = mf.by_group['selection_rate'].to_dict()
            priv_sr = sr_group.get(privileged_group, list(sr_group.values())[0])
            di_scores = {str(g): float(sr / priv_sr) if priv_sr > 0 else 0.0 for g, sr in sr_group.items()}
            
            model_results[name] = { "accuracy": float(acc), "disparate_impact": di_scores, "dp_ratio": float(demographic_parity_ratio(y_true, y_pred, sensitive_features=self.df[protected_attribute])) }
            if "Baseline" in name: selection_rates = {str(k): float(v) for k, v in sr_group.items()}

        # Auto-Selector Logic (Finds the best model passing the 80% legal threshold)
        fair_models = {k: v for k, v in model_results.items() if v["dp_ratio"] >= 0.80}
        if fair_models:
            best = max(fair_models, key=lambda k: fair_models[k]["accuracy"])
            reason = f"Selected because it passes compliance (DP={fair_models[best]['dp_ratio']:.2f}) with optimal accuracy ({fair_models[best]['accuracy']*100:.1f}%)."
        else:
            best = max(model_results, key=lambda k: model_results[k]["accuracy"])
            reason = f"CRITICAL WARNING: No models passed 80% parity. {best} is the most accurate but requires immediate Bias Mitigation."

        # Statistical Rigor test
        try:
            _, p_value, _, _ = chi2_contingency(pd.crosstab(self.df[protected_attribute], y_true))
        except: p_value = 1.0

        # Proxy Detective
        proxy_alerts = []
        for col in self.df.select_dtypes(include='number').columns:
            if col != target_column and col != protected_attribute:
                corr = abs(self.df[col].corr(pd.factorize(self.df[protected_attribute])[0]))
                if corr > 0.7: proxy_alerts.append({"feature": col, "correlation": float(corr)})

        chart_data = [{"name": str(g), "Original Rate %": round(r * 100, 2)} for g, r in selection_rates.items()]

        return {
            "statistical_significance": {"p_value": float(p_value), "is_reliable": bool(p_value <= 0.05)},
            "proxy_alerts": proxy_alerts,
            "model_comparison": model_results,
            "auto_selector": { "recommended_model": best, "reason": reason, "requires_mitigation": len(fair_models) == 0 },
            "chartData": chart_data,
        }

    def mitigate_bias(self, target_column: str, protected_attribute: str, favorable_class: str, privileged_group: str):
        X = pd.get_dummies(self.df.drop(columns=[target_column]))
        sensitive_cols = [c for c in X.columns if protected_attribute in c]
        
        cr = CorrelationRemover(sensitive_feature_ids=sensitive_cols)
        X_mitigated = cr.fit_transform(X)
        
        cols_kept = [c for c in X.columns if c not in sensitive_cols]
        df_mitigated = pd.DataFrame(X_mitigated, columns=cols_kept)
        df_mitigated[target_column] = self.df[target_column].values

        y_true = (self.df[target_column].astype(str) == favorable_class).astype(int)
        
        # We use Logistic Regression to verify the dataset is cleaned, as it is highly interpretable
        lr = LogisticRegression(max_iter=1000)
        lr.fit(df_mitigated.drop(columns=[target_column]).fillna(0), y_true)
        y_pred_clean = lr.predict(df_mitigated.drop(columns=[target_column]).fillna(0))

        mf_clean = MetricFrame(metrics={"selection_rate": selection_rate}, y_true=y_true, y_pred=y_pred_clean, sensitive_features=self.df[protected_attribute])
        clean_rates = {str(k): float(v) for k, v in mf_clean.by_group['selection_rate'].to_dict().items()}

        chart_data = [{"name": str(g), "Cleaned Rate %": round(r * 100, 2)} for g, r in clean_rates.items()]

        return { "message": "Mitigation applied.", "mitigated_features_count": len(cols_kept), "chartData": chart_data }