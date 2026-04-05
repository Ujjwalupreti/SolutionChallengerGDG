import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    false_negative_rate,
    demographic_parity_ratio
)
from fairlearn.preprocessing import CorrelationRemover
from scipy.stats import chi2_contingency

class FairnessAnalyzer:
    def __init__(self):
        self.df = None
        self._last_config = {}  # Store last audit config for mitigation re-use
        # Market-standard models for comparison
        self.models = {
            "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000),
            "Random Forest (Non-Linear)": RandomForestClassifier(n_estimators=100)
        }

    def load_data(self, df: pd.DataFrame):
        """Loads dataset into memory."""
        self.df = df

    def _get_intersectional_col(self, protected_attribute: str):
        """Creates an intersectional column if multiple attributes are provided."""
        attr_list = [attr.strip() for attr in protected_attribute.split(',')]
        if len(attr_list) > 1:
            col_name = "Intersectional_Group"
            self.df[col_name] = self.df[attr_list].astype(str).agg('_'.join, axis=1)
            return col_name, attr_list
        return attr_list[0], attr_list

    def evaluate_bias(self, target_column: str, protected_attribute: str, favorable_class: str, privileged_group: str, prediction_column: str = None):
        if self.df is None:
            raise ValueError("Data not loaded. Upload a CSV first.")
        
        # Store config for mitigate_bias to re-use
        self._last_config = {
            "target_column": target_column,
            "protected_attribute": protected_attribute,
            "favorable_class": favorable_class,
            "privileged_group": privileged_group,
        }

        protected_col, attr_list = self._get_intersectional_col(protected_attribute)
        
        # 1. Basic Demographic Stats
        total_records = len(self.df)
        demographic_counts = self.df[protected_col].value_counts().to_dict()
        demographic_percentages = {str(k): float(v / total_records) for k, v in demographic_counts.items()}
        
        # 2. Data Preparation
        y_true = (self.df[target_column].astype(str) == favorable_class).astype(int)
        X = pd.get_dummies(self.df.drop(columns=[target_column]))
        X = X.fillna(0)
        
        # 3. Multi-Model Evaluation & Selection Rates
        model_results = {}
        selection_rates = {}
        tpr_data = {}
        fpr_data = {}
        
        for name, model in self.models.items():
            model.fit(X, y_true)
            y_pred = model.predict(X)
            
            # Metrics Frame
            mf = MetricFrame(
                metrics={
                    "selection_rate": selection_rate,
                    "tpr": true_positive_rate,
                    "fpr": false_positive_rate,
                    "fnr": false_negative_rate
                },
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=self.df[protected_col]
            )
            
            # Disparate Impact Calculation
            sr_group = mf.by_group['selection_rate'].to_dict()
            priv_sr = sr_group.get(privileged_group, list(sr_group.values())[0])
            di_scores = {str(g): float(sr / priv_sr) if priv_sr > 0 else 0.0 for g, sr in sr_group.items()}
            
            model_results[name] = {
                "disparate_impact": di_scores,
                "dp_ratio": float(demographic_parity_ratio(y_true, y_pred, sensitive_features=self.df[protected_col]))
            }
            
            # Save rates for the primary Baseline model for the frontend charts
            if "Baseline" in name:
                selection_rates = {str(k): float(v) for k, v in sr_group.items()}
                tpr_data = {str(k): float(v) for k, v in mf.by_group['tpr'].to_dict().items()}
                fpr_data = {str(k): float(v) for k, v in mf.by_group['fpr'].to_dict().items()}

        # 4. Statistical Rigor (p-value)
        contingency_table = pd.crosstab(self.df[protected_col], y_true)
        try:
            _, p_value, _, _ = chi2_contingency(contingency_table)
        except:
            p_value = 1.0
            
        # 5. Proxy Variable Detective (Indirect Bias)
        proxy_alerts = []
        for col in self.df.columns:
            if col in [target_column, protected_col] or col in attr_list:
                continue
            try:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    corr = self.df[col].corr(self.df[protected_col] if pd.api.types.is_numeric_dtype(self.df[protected_col]) else self.df[protected_col].astype('category').cat.codes)
                    if abs(corr) > 0.7:
                        proxy_alerts.append({"feature": col, "correlation": float(abs(corr))})
            except:
                pass

        # 6. Build chartData array for Recharts ComposedChart
        chart_data = []
        for group, rate in selection_rates.items():
            chart_data.append({
                "name": str(group),
                "Original Rate %": round(rate * 100, 2),
            })

        return {
            "statistical_significance": {"p_value": float(p_value), "is_reliable": bool(p_value <= 0.05)},
            "proxy_alerts": proxy_alerts,
            "model_comparison": model_results,
            "chartData": chart_data,
            "selection_rate_by_group": selection_rates,
            "true_positive_rate_by_group": tpr_data,
            "false_positive_rate_by_group": fpr_data,
            "demographic_percentages": demographic_percentages,
            "disparate_impact": model_results["Logistic Regression (Baseline)"]["disparate_impact"]
        }

    def mitigate_bias(self, target_column: str, protected_attribute: str, favorable_class: str = None, privileged_group: str = None):
        """Uses CorrelationRemover to scrub bias, then re-evaluates to build chartData with Cleaned Rate %."""
        if self.df is None:
            raise ValueError("Data not loaded.")

        # Fall back to stored config if not passed explicitly
        if favorable_class is None:
            favorable_class = self._last_config.get("favorable_class", "1")
        if privileged_group is None:
            privileged_group = self._last_config.get("privileged_group", "")

        protected_col, attr_list = self._get_intersectional_col(protected_attribute)

        X = pd.get_dummies(self.df.drop(columns=[target_column]))
        # Identify sensitive columns in the one-hot encoded dataframe
        sensitive_cols = [c for c in X.columns if any(attr.strip() in c for attr in protected_attribute.split(','))]
        
        cr = CorrelationRemover(sensitive_feature_ids=sensitive_cols)
        X_mitigated = cr.fit_transform(X)
        
        # Reconstruct DataFrame (excluding the original sensitive features)
        cols_kept = [c for c in X.columns if c not in sensitive_cols]
        df_mitigated = pd.DataFrame(X_mitigated, columns=cols_kept)
        df_mitigated[target_column] = self.df[target_column].values

        # Re-run baseline model on cleaned data to get post-mitigation rates
        y_true = (self.df[target_column].astype(str) == favorable_class).astype(int)
        X_clean = df_mitigated.drop(columns=[target_column]).fillna(0)

        try:
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X_clean, y_true)
            y_pred_clean = lr.predict(X_clean)

            mf_clean = MetricFrame(
                metrics={"selection_rate": selection_rate},
                y_true=y_true,
                y_pred=y_pred_clean,
                sensitive_features=self.df[protected_col]
            )
            clean_rates = {str(k): float(v) for k, v in mf_clean.by_group['selection_rate'].to_dict().items()}
        except Exception:
            clean_rates = {}

        # Re-fetch original rates from stored chartData context
        orig_rates = {}
        try:
            y_pred_orig = list(self.models.values())[0].predict(X.fillna(0))
            mf_orig = MetricFrame(
                metrics={"selection_rate": selection_rate},
                y_true=y_true,
                y_pred=y_pred_orig,
                sensitive_features=self.df[protected_col]
            )
            orig_rates = {str(k): float(v) for k, v in mf_orig.by_group['selection_rate'].to_dict().items()}
        except Exception:
            pass

        # Build combined chartData with both Original and Cleaned rates
        all_groups = set(list(orig_rates.keys()) + list(clean_rates.keys()))
        chart_data = []
        for group in all_groups:
            entry: dict = {"name": str(group)}
            if group in orig_rates:
                entry["Original Rate %"] = round(orig_rates[group] * 100, 2)
            if group in clean_rates:
                entry["Cleaned Rate %"] = round(clean_rates[group] * 100, 2)
            chart_data.append(entry)

        return {
            "message": "Mitigation applied successfully.",
            "mitigated_features_count": len(cols_kept),
            "chartData": chart_data,
            "preview": df_mitigated.head(10).to_dict(orient="records")
        }