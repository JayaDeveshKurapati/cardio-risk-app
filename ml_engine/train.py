import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib

# Core SciKit-Learn Framework Components
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline

# Advanced Gradient Tree Boosting Frameworks
import xgboost as xgb
import lightgbm as lgb
import catboost as cb  # Fixed the import typo completely here

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def build_and_train_pipeline(csv_path="../cardio_train.csv"):
    print("=" * 70)
    print("[1/5] STEP 1 & 2 – LOADING DATA & PREPROCESSING")
    print("=" * 70)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing dataset! Place 'cardio_train.csv' in the root directory: {os.path.abspath(csv_path)}")
        
    df = pd.read_csv(csv_path, sep=";")
    print(f" Raw shape verified: {df.shape}")
    df.drop(columns=["id"], inplace=True, errors='ignore')

    # Convert days to years mapping
    df["age"] = df["age"] / 365.25
    
    # Missing Value Handling Imputation
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())
        
    # Z-Score Outlier filtering set to 4 Standard Deviations
    for col in ["age", "height", "weight", "ap_hi", "ap_lo"]:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df = df[np.abs(df[col] - mean_val) <= 4 * std_val]
        
    # Apply baseline physiological bounding sanity check
    df = df[df["ap_hi"] >= df["ap_lo"]]
    print(f" Shape after outlier processing: {df.shape}")

    print("\n" + "=" * 70)
    print("[2/5] STEP 3, 4 & 5 – ENGINEERING, NORMALISATION & BALANCING")
    print("=" * 70)
    
    # Strip hidden whitespace from columns to avoid indexing errors
    df.columns = df.columns.str.strip()
    
    # Map 'target' to 'cardio' to match your column layout
    if 'target' in df.columns:
        df.rename(columns={'target': 'cardio'}, inplace=True)
    
    # Double-check safety switch
    if 'cardio' not in df.columns:
        raise KeyError(f"Could not find 'cardio' or 'target' column. Available columns: {list(df.columns)}")

    # Feature Engineering (Exact equations matching your version)
    df["bmi"] = df["weight"] / (df["height"] / 100.0) ** 2
    df = df[(df["bmi"] > 10) & (df["bmi"] <= 60)]
    df["chol_gluc_ratio"] = df["cholesterol"] / df["gluc"]
    df["ap_ratio"] = df["ap_hi"] * df["ap_lo"]  
    df["gender"] = df["gender"] - 1              

    X = df.drop(columns=["cardio"])
    y = df["cardio"].reset_index(drop=True)

    # Min-Max Normalisation
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Dual-Stage Class Resampling Engine Execution
    over = SMOTE(random_state=42)
    under = RandomUnderSampler(random_state=42)
    X_res, y_res = over.fit_resample(X_scaled, y)
    X_res, y_res = under.fit_resample(X_res, y_res)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.20, random_state=42, stratify=y_res
    )
    
    X_train = pd.DataFrame(X_train, columns=X.columns).reset_index(drop=True)
    X_test = pd.DataFrame(X_test, columns=X.columns).reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)
    print(f" Finalized Dataset Shapes: X={X_train.shape}, y={y_train.shape}")

    print("\n" + "=" * 70)
    print("[3/5] STEP 7 & 8 – BASE MODELING & OUT-OF-FOLD STACKING")
    print("=" * 70)
    
    gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    cat_model = cb.CatBoostClassifier(iterations=200, depth=6, learning_rate=0.05, verbose=0, random_seed=42)
    lgb_model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, verbose=-1, random_state=42, n_jobs=-1)
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    
    # Fast SVM pipeline configuration
    approx_rbf = Nystroem(gamma=0.2, random_state=42, n_components=300)
    svm_fast_rbf = make_pipeline(approx_rbf, LinearSVC(C=1.0, max_iter=2000, random_state=42))
    svm_model = CalibratedClassifierCV(svm_fast_rbf, cv=3)
    
    nn_model = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", solver="adam", alpha=0.001, max_iter=300, random_state=42)
    
    base_models = [
        ("Gradient Boosting", gb_model), ("CatBoost", cat_model), ("LightGBM", lgb_model),
        ("Random Forest", rf_model), ("SVM", svm_model), ("Neural Network", nn_model),
    ]

    n_base = len(base_models)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_train = np.zeros((X_train.shape[0], n_base))
    meta_test = np.zeros((X_test.shape[0], n_base))
    
    for j, (name, model) in enumerate(base_models):
        print(f" [{j+1}/{n_base}] Processing out-of-fold feature mapping: {name}...")
        fold_test = np.zeros((X_test.shape[0], skf.n_splits))
        for k, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            meta_train[val_idx, j] = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            fold_test[:, k] = model.predict_proba(X_test)[:, 1]
        meta_test[:, j] = fold_test.mean(axis=1)

    meta_train_full = np.hstack([meta_train, X_train.values])
    meta_test_full = np.hstack([meta_test, X_test.values])

    print("\n[4/5] Training final XGBoost Meta-Model layers...")
    xgb_meta = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, eval_metric="logloss", random_state=42, n_jobs=-1)
    xgb_meta.fit(meta_train_full, y_train)
    
    final_proba = xgb_meta.predict_proba(meta_test_full)[:, 1]
    final_preds = (final_proba >= 0.50).astype(int)

    print("\n" + "=" * 75)
    print("[5/5] PRINTING MODEL METRICS")
    print("=" * 75)
    acc = accuracy_score(y_test, final_preds)
    prec = precision_score(y_test, final_preds)
    rec = recall_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)
    auc = roc_auc_score(y_test, final_proba)
    
    print(f" 📊 Hybrid Model Accuracy  : {acc:>6.1%}  | Target: 82.0%")
    print(f" 🎯 Hybrid Model Precision : {prec:>6.1%}  | Target: 81.0%")
    print(f" 🧪 Hybrid Model Recall    : {rec:>6.1%}  | Target: 83.0%")
    print(f" 🧬 Hybrid Model F1-Score  : {f1:>6.1%}  | Target: 82.0%")
    print(f" 📈 Hybrid Model AUC-ROC   : {auc:>6.3f}  | Target: 0.820")
    print("=" * 75 + "\n")
    
    # Save the deployment artifacts directly into your backend workspace
    os.makedirs("../backend", exist_ok=True)
    
    artifacts = {
        "base_models": base_models,
        "meta_model": xgb_meta,
        "feature_columns": list(X.columns)
    }
    joblib.dump(artifacts, "../backend/ensemble_model_metadata.pkl")
    joblib.dump(scaler, "../backend/scaler.pkl")
    print("✨ Complete! Matched pipeline weights saved directly to your backend folder.")

if __name__ == "__main__":
    build_and_train_pipeline()
