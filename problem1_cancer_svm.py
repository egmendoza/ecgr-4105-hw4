"""Problem 1: SVM classifier for breast cancer (malignant vs. benign) with different kernels,
compared against the logistic regression models from Homework 3."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)

SEED = 42
KERNELS = ["linear", "poly", "rbf", "sigmoid"]
os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/Cancer.csv")
# Drop the ID column and the empty trailing column produced by a trailing comma.
df = df.drop(columns=[c for c in df.columns if c == "id" or c.startswith("Unnamed")])
y = (df["diagnosis"] == "M").astype(int).values   # Malignant = 1 (positive), Benign = 0
X = df.drop(columns="diagnosis")
assert X.shape[1] == 30

# Same split as Homework 3 (80/20, stratified, seed 42) so the test set is identical.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
scaler = StandardScaler().fit(X_train)             # fit on training data only
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)


def metrics(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


print(f"Train/test sizes: {len(y_train)}/{len(y_test)}")

# ---------------- Each kernel at the default setting (C = 1, gamma = "scale") ----------------
rows, cms = {}, {}
for k in KERNELS:
    model = SVC(kernel=k, C=1.0, gamma="scale").fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    rows[f"SVM ({k})"] = metrics(y_test, pred)
    cms[k] = confusion_matrix(y_test, pred)
    print(f"\nSVM kernel = {k} (C = 1)")
    print(f"  train acc {model.score(X_train_s, y_train):.4f}, support vectors {model.n_support_.sum()}")
    print("  test metrics:", {m: round(v, 4) for m, v in rows[f'SVM ({k})'].items()})
    print("  confusion [[TN, FP], [FN, TP]]:", cms[k].tolist())

# Homework 3 results (my gradient-descent logistic regression on the same split and seed),
# copied from results_problem2.txt of Homework 3.
rows["LogReg, no penalty (HW3)"] = {"accuracy": 0.9737, "precision": 0.9756, "recall": 0.9524, "f1": 0.9639}
rows["LogReg, L2 lambda=3 (HW3)"] = {"accuracy": 0.9825, "precision": 1.0, "recall": 0.9524, "f1": 0.9756}
res = pd.DataFrame(rows).T
print("\nSummary (test set):\n", res.round(4).to_string())

ax = res[["accuracy", "precision", "recall"]].plot(kind="bar", figsize=(10, 4.8), width=0.8)
ax.set(ylim=(0.65, 1.005), ylabel="Score (test set)",
       title="Problem 1: SVM kernels vs. logistic regression (C = 1)")
ax.set_xticklabels(res.index, rotation=20, ha="right")
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/p1_kernel_comparison.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
for a, k in zip(axes, KERNELS):
    ConfusionMatrixDisplay(cms[k], display_labels=["Benign (0)", "Malignant (1)"]).plot(
        ax=a, cmap="Blues", colorbar=False)
    a.set_title(f"{k} kernel", fontsize=10)
fig.suptitle("Problem 1: Confusion matrices (test set, C = 1)")
fig.tight_layout()
fig.savefig("figures/p1_confusion.png", dpi=150)
plt.close(fig)

# ---------------- Effect of the regularization parameter C ----------------
Cs = np.logspace(-2, 3, 11)
names = ["accuracy", "precision", "recall"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
print("\nBest C per kernel (by test accuracy):")
for k in KERNELS:
    sweep = pd.DataFrame(
        [metrics(y_test, SVC(kernel=k, C=c, gamma="scale").fit(X_train_s, y_train).predict(X_test_s))
         for c in Cs], index=Cs)
    for a, m in zip(axes, names):
        a.semilogx(Cs, sweep[m], marker="o", label=k)
    best = sweep["accuracy"].idxmax()
    print(f"  {k:8s} C = {best:<7.3g} " + str({m: round(sweep.loc[best, m], 4) for m in names}))
for a, m in zip(axes, names):
    a.set(title=m.capitalize(), xlabel="C")
    a.grid(alpha=0.3)
axes[0].set_ylabel("Score (test set)")
axes[0].legend()
fig.suptitle("Problem 1: Test metrics vs. C for each kernel")
fig.tight_layout()
fig.savefig("figures/p1_C_sweep.png", dpi=150)
plt.close(fig)
