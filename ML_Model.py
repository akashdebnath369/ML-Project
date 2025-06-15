import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.semi_supervised import SelfTrainingClassifier

##############################################################################
# 1)  CONFIGURATION (↓↓ change only these two lines if you move the dataset) #
##############################################################################
# CSV_PATH     = Path(_file_).with_name("Book1data.csv")  # same dir as script
# When running in a notebook, _file_ is not defined.
# We can assume the CSV is in the current working directory.
CSV_PATH     = Path("Book1data.csv")
LABEL_FRAC   = 0.30       # proportion of the training pool initially labelled
##############################################################################

assert CSV_PATH.exists(), f"CSV file not found at: {CSV_PATH.resolve()}"

###########################################################################
# 2)  LOAD + LABEL CONSTRUCTION
###########################################################################
df = pd.read_csv(CSV_PATH)

protected_tags = {
    "gender", "religion", "caste", "race", "sexual_orientation", "nationality"
}
df["hate"] = (
    df["Target"].str.strip().str.lower().isin(protected_tags)
).astype(int)

###########################################################################
# 3)  FEATURE ENCODING
###########################################################################
feature_cols = ["Sentiment", "Sarcasm", "Vulgar", "Abuse"]
X_raw = df[feature_cols]
y     = df["hate"].values

ohe = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols)],
    remainder="drop"
)
X = ohe.fit_transform(X_raw)

###########################################################################
# 4)  SPLIT AND MASK LABELS FOR SEMI-SUPERVISION
###########################################################################
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

n_labeled     = int(np.ceil(LABEL_FRAC * X_train_full.shape[0]))
rng           = np.random.RandomState(42)
indices       = rng.permutation(X_train_full.shape[0])
labeled_set   = indices[:n_labeled]
unlabeled_set = indices[n_labeled:]

y_train_semi = y_train_full.copy()
y_train_semi[unlabeled_set] = -1          # -1 = unknown label

###########################################################################
# 5)  SELF-TRAINING WRAPPER
###########################################################################
base_clf = GradientBoostingClassifier(random_state=42)

model = SelfTrainingClassifier(
    base_estimator = base_clf,
    threshold      = 0.8,
    criterion      = "threshold",
    max_iter       = 20,
    verbose        = True
)
model.fit(X_train_full, y_train_semi)

###########################################################################
# 6)  EVALUATION
###########################################################################
print("\n=== Held-out test performance ===")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

###########################################################################
# 7)  SIMPLE INFERENCE HELPER
###########################################################################
def predict_hate(sentiment, sarcasm, vulgar, abuse):
    """
    Predict 0 = non-hate, 1 = hate  for a single instance
    >>> predict_hate("Negative", "Sarcastic", "Vulgar", "Abusive")
    """
    # Convert the input list to a pandas DataFrame with the correct columns
    X_new_df = pd.DataFrame([[sentiment, sarcasm, vulgar, abuse]], columns=feature_cols)
    X_new = ohe.transform(X_new_df)
    return int(model.predict(X_new)[0])


# QUICK DEMO
if __name__ == "_main_":
    demo = ["Negative", "Sarcastic", "Vulgar", "Abusive"]
    print(f"\nDemo instance {demo} → hate? {predict_hate(*demo)}")