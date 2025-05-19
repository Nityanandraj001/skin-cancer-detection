import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────────────────────────────────────────────
# 1. Helper to load TXT-files
def load_data(filepath):
    freq, s_val = np.loadtxt(filepath, usecols=(0, 1), unpack=True)
    return pd.DataFrame({"Frequency": freq, "S_parameter": s_val})

# 2. Read datasets & add labels
cancer_df  = load_data("skincancerwithtumor.txt")
healthy_df = load_data("skincancerwithoutcancer.txt")

cancer_df["Label"]  = 1      # 1 → skin-cancer
healthy_df["Label"] = 0      # 0 → healthy / normal

# 3. Combine & shuffle
full_df = pd.concat([cancer_df, healthy_df], ignore_index=True)
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. Train / test split
X = full_df[["Frequency", "S_parameter"]]
y = full_df["Label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------------------------------------------------------
# 6. Predict on the WHOLE data set (for smoother curves)
full_df["Predicted"] = model.predict(X)

# 7. Aggregate curves: mean S11 per frequency per class
grouped = (
    full_df.groupby(["Label", "Frequency"])["S_parameter"]
    .mean()
    .reset_index()
)

# 8. Split into two DataFrames for plotting
normal_curve  = grouped[grouped["Label"] == 0]
cancer_curve  = grouped[grouped["Label"] == 1]

# 9. Plot – mimic the reference style
plt.figure(figsize=(7, 4))
plt.plot(
    normal_curve["Frequency"],
    normal_curve["S_parameter"],
    linewidth=2.5,
    label="Normal",
    color="black",
)
plt.plot(
    cancer_curve["Frequency"],
    cancer_curve["S_parameter"],
    linewidth=2.5,
    label="Skin Cancer",
    color="red",
)

plt.gca().invert_yaxis()                 # dB axis points downward
plt.xlim(full_df["Frequency"].min(), full_df["Frequency"].max())
plt.ylim(-60, 5)                         # match reference y-range
plt.xlabel("Frequency (GHz)", fontsize=12)
plt.ylabel("S₁₁ (dB)", fontsize=12)
plt.title("S₁₁ Response – Normal vs Skin Cancer", fontsize=14)
plt.legend(frameon=False, fontsize=11)
plt.tight_layout()
plt.show()

# 10. Standard metrics (unchanged)
print("Classification Report:\n", classification_report(y_test, model.predict(X_test)))
print("Confusion Matrix:\n", confusion_matrix(y_test, model.predict(X_test)))
