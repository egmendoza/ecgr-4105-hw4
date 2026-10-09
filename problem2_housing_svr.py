"""Problem 2: SVR model for housing price with different kernels,
compared against linear regression with an L2 parameter penalty."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

SEED = 42
KERNELS = ["linear", "poly", "rbf", "sigmoid"]
FEATURES = ["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement",
            "hotwaterheating", "airconditioning", "parking", "prefarea"]
BINARY = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/Housing.csv")
df[BINARY] = df[BINARY].apply(lambda c: c.map({"yes": 1, "no": 0}))
X, y = df[FEATURES].values.astype(float), df["price"].values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
# Standardize inputs AND the target (fit on training data only). SVR with C = 1 and
# epsilon = 0.1 cannot fit a target measured in millions, so the price is scaled for
# training and every prediction is converted back before computing metrics.
xs = StandardScaler().fit(X_train)
ys = StandardScaler().fit(y_train.reshape(-1, 1))
X_train_s, X_test_s = xs.transform(X_train), xs.transform(X_test)
y_train_s = ys.transform(y_train.reshape(-1, 1)).ravel()


def to_price(p):
    return ys.inverse_transform(p.reshape(-1, 1)).ravel()


def evaluate(model):
    p_train, p_test = to_price(model.predict(X_train_s)), to_price(model.predict(X_test_s))
    return {"train_R2": r2_score(y_train, p_train), "test_R2": r2_score(y_test, p_test),
            "test_RMSE": np.sqrt(mean_squared_error(y_test, p_test)),
            "test_MAE": mean_absolute_error(y_test, p_test)}, p_test


print(f"Train/test sizes: {len(y_train)}/{len(y_test)}")

# ---------------- Each kernel at the default setting (C = 1, epsilon = 0.1) ----------------
rows, preds, models = {}, {}, {}
for k in KERNELS:
    models[k] = SVR(kernel=k, C=1.0, epsilon=0.1, gamma="scale").fit(X_train_s, y_train_s)
    rows[f"SVR ({k})"], preds[f"SVR ({k})"] = evaluate(models[k])
    print(f"SVR kernel = {k}: support vectors {len(models[k].support_)} of {len(y_train)}")
# Baseline: linear regression with an L2 parameter penalty (ridge) on the same inputs and split.
rows["Linear reg. + L2 penalty"], preds["Linear reg. + L2 penalty"] = evaluate(
    Ridge(alpha=1.0).fit(X_train_s, y_train_s))
res = pd.DataFrame(rows).T
print("\nSummary:\n", res.round(4).to_string())

# Regression model plot in the style of the course SVR sample: data plus the fitted curve
# of each kernel against one input (area), other inputs held at their training mean.
area = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
grid = np.tile(X_train.mean(axis=0), (200, 1))
grid[:, 0] = area
plt.figure(figsize=(8, 5))
plt.scatter(X_test[:, 0], y_test / 1e6, s=15, c="gray", alpha=0.5, label="test data")
for k in KERNELS:
    plt.plot(area, to_price(models[k].predict(xs.transform(grid))) / 1e6, lw=2, label=f"{k} model")
plt.xlabel("Area")
plt.ylabel("Price (millions)")
plt.title("Problem 2: Support Vector Regression, price vs. area")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/p2_svr_vs_area.png", dpi=150)
plt.close()

# Predicted vs. actual price on the test set
fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
axes = axes.ravel()
axes[5].axis("off")
lim = [y_test.min() / 1e6 * 0.9, y_test.max() / 1e6 * 1.05]
for a, (name, p) in zip(axes, preds.items()):
    a.scatter(y_test / 1e6, p / 1e6, s=18, alpha=0.7)
    a.plot(lim, lim, "r--", lw=1, label="ideal")
    a.set_title(f"{name}, $R^2$ = {rows[name]['test_R2']:.3f}")
    a.set_xlabel("Actual price (millions)")
    a.tick_params(labelbottom=True)
    a.grid(alpha=0.3)
axes[0].set_ylabel("Predicted price (millions)")
axes[3].set_ylabel("Predicted price (millions)")
axes[0].legend()
fig.tight_layout()
fig.savefig("figures/p2_pred_vs_actual.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
res["test_R2"].plot(kind="bar", ax=axes[0])
axes[0].set_title("Test $R^2$ (higher is better)")
(res["test_RMSE"] / 1e6).plot(kind="bar", ax=axes[1], color="C1")
axes[1].set_title("Test RMSE, millions (lower is better)")
for a in axes:
    a.set_xticklabels(res.index, rotation=20, ha="right")
    a.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("figures/p2_kernel_comparison.png", dpi=150)
plt.close(fig)

# ---------------- Effect of the regularization parameter C ----------------
Cs = np.logspace(-2, 3, 11)
plt.figure(figsize=(8, 4.5))
print("\nBest C per kernel (by test R2):")
for k in KERNELS:
    r2 = [evaluate(SVR(kernel=k, C=c, epsilon=0.1, gamma="scale").fit(X_train_s, y_train_s))[0]["test_R2"]
          for c in Cs]
    plt.semilogx(Cs, r2, marker="o", label=k)
    print(f"  {k:8s} C = {Cs[int(np.argmax(r2))]:<7.3g} test R2 = {max(r2):.4f}")
plt.ylim(-0.5, 0.8)
plt.xlabel("C")
plt.ylabel("Test $R^2$")
plt.title("Problem 2: Test $R^2$ vs. C for each kernel")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/p2_C_sweep.png", dpi=150)
plt.close()
