# train_best_model.py
import json
import os
import warnings
import numpy as np
import pandas as pd
from pprint import pprint

# Embeddings
from sentence_transformers import SentenceTransformer

# Models & utils
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Optional libs (not required)
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# -------------------------
# Helper functions
# -------------------------
def choose_label_column(df):
    """Prefer obvious label columns if present; otherwise check 'rating' validity; else return None."""
    preferred = ['label', 'target', 'cuisine', 'category', 'meal_type', 'rating']
    for c in preferred:
        if c in df.columns:
            # Check this is usable as a classification label
            nunique = df[c].nunique(dropna=True)
            if nunique >= 2:
                return c
    return None

def create_pseudo_labels(embeddings, n_clusters=4):
    """Create pseudo-labels using KMeans clustering on embeddings (if no true labels available)."""
    from sklearn.cluster import KMeans
    n_clusters = min(n_clusters, max(2, embeddings.shape[0] // 5))  # avoid too many clusters for tiny data
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels.astype(str)

def safe_cv_strategy(y, max_splits=5):
    """Return an appropriate CV splitter (StratifiedKFold if possible, else KFold)."""
    counts = pd.Series(y).value_counts()
    min_count = counts.min()
    if min_count >= 2:
        n_splits = min(max_splits, int(min_count))
        if n_splits < 2:
            n_splits = 2
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        # fallback to KFold (not stratified) if there's a class with a single sample
        n_splits = 2
        return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

# -------------------------
# 1. Load dataset
# -------------------------
DATA_PATH = "data.json"
assert os.path.exists(DATA_PATH), f"{DATA_PATH} not found. Put your dataset as 'data.json' in cwd."

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data_list = json.load(f)
data = pd.DataFrame(data_list)
print("✅ Loaded dataset:", data.shape)
print("Columns:", list(data.columns))

# -------------------------
# 2. Preprocess text
# -------------------------
def to_text_field(row):
    name = row.get('recipe_name', '') or ''
    ing = row.get('ingredients', '')
    steps = row.get('steps', '')
    if isinstance(ing, list):
        ing = ", ".join(map(str, ing))
    else:
        ing = str(ing)
    if isinstance(steps, list):
        steps = " ".join(map(str, steps))
    else:
        steps = str(steps)
    return f"{name} {ing} {steps}"

data['text'] = data.apply(to_text_field, axis=1)
print("Sample text (first row):\n", data['text'].iloc[0][:400])

# -------------------------
# 3. Compute / load embeddings
# -------------------------
EMB_PATH = "bert_embeddings.npy"

if os.path.exists(EMB_PATH):
    recipe_embeddings = np.load(EMB_PATH)
    print("✅ Loaded existing embeddings from", EMB_PATH)
else:
    # try a stronger model first; fall back to smaller if not available or memory constrained
    model_names = ['all-mpnet-base-v2', 'all-MiniLM-L6-v2']
    bert_model = None
    for mname in model_names:
        try:
            print("🔄 Loading SentenceTransformer:", mname)
            bert_model = SentenceTransformer(mname)
            print("✅ Using embedding model:", mname)
            break
        except Exception as e:
            print("⚠️ Failed to load", mname, ":", e)
            continue
    if bert_model is None:
        raise RuntimeError("No sentence-transformers model could be loaded. Install sentence-transformers and try again.")
    recipe_texts = data['text'].tolist()
    recipe_embeddings = bert_model.encode(recipe_texts, show_progress_bar=True, convert_to_numpy=True)
    np.save(EMB_PATH, recipe_embeddings)
    print("✅ Saved embeddings to", EMB_PATH)

# -------------------------
# 4. Determine label (or create pseudo-labels)
# -------------------------
label_col = choose_label_column(data)
if label_col is not None:
    y = data[label_col].astype(str).values
    print(f"✅ Using existing label column: '{label_col}' with {len(np.unique(y))} classes.")
else:
    # create pseudo-labels using KMeans
    print("ℹ️ No suitable label column found. Creating pseudo-labels with clustering (unsupervised)...")
    n_clusters = 4
    y = create_pseudo_labels(recipe_embeddings, n_clusters=n_clusters)
    print("✅ Created pseudo-labels. Number of pseudo-classes:", len(np.unique(y)))

# Print label distribution
print("Label distribution:")
print(pd.Series(y).value_counts())

# -------------------------
# 5. Dimensionality reduction (optional)
# For small datasets using full embeddings is fine; if very high-dim, you can reduce.
# -------------------------
# If you want to reduce dims, uncomment the block below.
# from sklearn.decomposition import TruncatedSVD
# svd = TruncatedSVD(n_components=min(128, recipe_embeddings.shape[1]-1), random_state=RANDOM_STATE)
# recipe_embeddings = svd.fit_transform(recipe_embeddings)
# print("Reduced embeddings shape:", recipe_embeddings.shape)

# -------------------------
# 6. Define candidate models
# -------------------------
models = {}

# Logistic (needs scaling)
models['LogisticRegression'] = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE))
])

# SVM
models['SVC'] = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE))
])

# Random Forest
models['RandomForest'] = RandomForestClassifier(
    n_estimators=250, max_depth=None, class_weight='balanced', random_state=RANDOM_STATE
)

# XGBoost (if available)
if HAS_XGB:
    models['XGBoost'] = xgb.XGBClassifier(
        n_estimators=200, use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE
    )

# LightGBM (if available)
if HAS_LGB:
    models['LightGBM'] = lgb.LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE)

print("Models to evaluate:", list(models.keys()))

# -------------------------
# 7. Cross-validated evaluation
# -------------------------
cv = safe_cv_strategy(y, max_splits=5)
results = []

for name, estimator in models.items():
    try:
        scores = cross_val_score(estimator, recipe_embeddings, y, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        results.append((name, mean_score, std_score))
        print(f"{name:12s} | Acc (cv mean ± std): {mean_score:.4f} ± {std_score:.4f}")
    except Exception as e:
        print(f"⚠️ Skipping {name} due to error during cross_val_score: {e}")

# Sort and show
results = sorted(results, key=lambda x: x[1], reverse=True)
print("\n== Cross-validated accuracy ranking ==")
for name, mean_score, std_score in results:
    print(f"{name:12s} : {mean_score:.4f} ± {std_score:.4f}")

if len(results) == 0:
    raise RuntimeError("No models could be evaluated. Check data & installed libraries.")

best_name, best_mean, best_std = results[0]
best_estimator = models[best_name]
print(f"\n🎉 Best model by CV accuracy: {best_name} ({best_mean:.4f} ± {best_std:.4f})")

# -------------------------
# 8. Cross-validated predictions & final metrics
# -------------------------
print("\n🔁 Generating cross-validated predictions (out-of-fold) for final metrics...")
y_pred_oof = cross_val_predict(best_estimator, recipe_embeddings, y, cv=cv, n_jobs=-1)

acc = accuracy_score(y, y_pred_oof)
print(f"\n🎯 Cross-validated Accuracy (overall, out-of-fold): {acc:.4f}\n")

print("📊 Classification Report (out-of-fold):")
print(classification_report(y, y_pred_oof, zero_division=0))

cm = confusion_matrix(y, y_pred_oof, labels=np.unique(y))
print("\n🧮 Confusion Matrix (rows=actual, cols=predicted):")
print(cm)

# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=np.unique(y), yticklabels=np.unique(y), cmap="Blues")
plt.title(f"Confusion Matrix - {best_name} (cross-validated, OOF)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -------------------------
# 9. Optionally run a quick RandomizedSearch on the best model (safeguarded)
# -------------------------
DO_HYPERPARAM_TUNING = True
if DO_HYPERPARAM_TUNING:
    print("\n⚙️ Running a short RandomizedSearchCV on the best model (limited iterations)...")
    # Only tune for certain known models
    param_distributions = None
    estimator_for_search = None

    if best_name == 'RandomForest':
        estimator_for_search = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
        param_distributions = {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 10, 20, 40],
            "min_samples_split": [2, 5, 10]
        }
    elif best_name == 'LogisticRegression':
        estimator_for_search = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, max_iter=5000))
        ])
        param_distributions = {
            "clf__C": [0.01, 0.1, 1, 10, 100]
        }
    elif best_name == 'XGBoost' and HAS_XGB:
        estimator_for_search = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE)
        param_distributions = {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.05, 0.1]
        }
    elif best_name == 'LightGBM' and HAS_LGB:
        estimator_for_search = lgb.LGBMClassifier(random_state=RANDOM_STATE)
        param_distributions = {
            "n_estimators": [100, 200, 400],
            "max_depth": [ -1, 10, 20],
            "learning_rate": [0.01, 0.05, 0.1]
        }
    else:
        print("No tuned hyperparam search configured for", best_name)

    if estimator_for_search is not None and param_distributions is not None:
        n_iter = 16
        try:
            rs = RandomizedSearchCV(estimator_for_search, param_distributions, n_iter=n_iter,
                                    scoring='accuracy', cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=1)
            rs.fit(recipe_embeddings, y)
            print("Best params from RandomizedSearchCV:")
            pprint(rs.best_params_)
            print("Best cross-val accuracy:", rs.best_score_)
            best_estimator = rs.best_estimator_
        except Exception as e:
            print("⚠️ Hyperparam tuning failed or too expensive:", e)

# -------------------------
# 10. Fit best estimator on full data and save model
# -------------------------
print(f"\n🔐 Fitting best model ({best_name}) on the full dataset and saving to disk...")
try:
    best_estimator.fit(recipe_embeddings, y)
    joblib.dump(best_estimator, "best_model.joblib")
    print("✅ Saved best model to best_model.joblib")
except Exception as e:
    print("⚠️ Failed to fit & save best estimator:", e)

# Save embeddings & label mapping for later use
np.save("bert_embeddings.npy", recipe_embeddings)
pd.Series(y, name='label').to_csv("labels.csv", index=False)
print("✅ Saved embeddings and label file.")

print("\n🎯 DONE. Summary:")
print(f" - Best model: {best_name}")
print(f" - Cross-validated OOF accuracy: {acc:.4f}")
print(" - Classification report printed above and confusion matrix plotted.")
print("\nNext suggestions to squeeze more accuracy:")
print(" - Use a real, meaningful label (cuisine / meal_type / difficulty) instead of random rating.")
print(" - Increase dataset size (200+ samples per class for robust results).")
print(" - If you have >1000 labeled examples, fine-tune a transformer classifier end-to-end.")
print(" - Consider feature engineering (ingredients as bag-of-words, metadata).")
