import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_regression

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(layout="wide")
st.title("Gradient Descent – Line Movement Only")

st.markdown("""
This visualization shows **only the regression line moving**
after each Gradient Descent update.
The **data points never move**.
""")

# ---------------------------------
# Sidebar controls
# ---------------------------------
st.sidebar.header("Controls")

gd_type = st.sidebar.selectbox(
    "Gradient Descent Type",
    ["Batch", "Stochastic", "Mini-Batch"]
)

lr = st.sidebar.slider("Learning Rate (η)", 0.0001, 0.05, 0.01)
epochs = st.sidebar.slider("Epochs", 1, 100, 30)

batch_size = st.sidebar.slider(
    "Mini-Batch Size",
    2, 20, 10,
    disabled=(gd_type != "Mini-Batch")
)

st.sidebar.subheader("Initial Line")
m = st.sidebar.slider("Initial slope (m)", -50.0, 50.0, -30.0)
b = st.sidebar.slider("Initial intercept (b)", -200.0, 200.0, -100.0)

run = st.sidebar.button("▶ Run Gradient Descent")

# ---------------------------------
# Dataset (fixed)
# ---------------------------------
X, y = make_regression(
    n_samples=100,
    n_features=1,
    n_informative=1,
    n_targets=1,
    noise=20,
    random_state=13
)

X = X.flatten()
y = y.flatten()
n = len(X)

# ---------------------------------
# FIXED AXIS LIMITS (CRITICAL)
# ---------------------------------
x_min, x_max = X.min() - 1, X.max() + 1
y_min, y_max = y.min() - 50, y.max() + 50

# ---------------------------------
# Loss function
# ---------------------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# ---------------------------------
# Placeholders
# ---------------------------------
plot_area = st.empty()
loss_area = st.empty()
loss_history = []

# ---------------------------------
# Gradient Descent Animation
# ---------------------------------
# ---------------------------------
# Gradient Descent Animation (FIXED)
# ---------------------------------
if run:

    step = 0  # visual step counter

    while step < epochs:

        # ---------- BATCH ----------
        if gd_type == "Batch":
            y_hat = m * X + b
            dm = (-2/n) * np.sum(X * (y - y_hat))
            db = (-2/n) * np.sum(y - y_hat)
            m -= lr * dm
            b -= lr * db
            step += 1

        # ---------- STOCHASTIC ----------
        elif gd_type == "Stochastic":
            i = np.random.randint(0, n)   # ONE POINT ONLY
            xi, yi = X[i], y[i]
            y_hat = m * xi + b
            dm = -2 * xi * (yi - y_hat)
            db = -2 * (yi - y_hat)
            m -= lr * dm
            b -= lr * db
            step += 1

        # ---------- MINI-BATCH ----------
        elif gd_type == "Mini-Batch":
            idx = np.random.choice(n, batch_size, replace=False)
            Xb, yb = X[idx], y[idx]
            y_hat = m * Xb + b
            dm = (-2/len(Xb)) * np.sum(Xb * (yb - y_hat))
            db = (-2/len(Xb)) * np.sum(yb - y_hat)
            m -= lr * dm
            b -= lr * db
            step += 1

        # ---------- LOSS ----------
        loss = mse(y, m * X + b)
        loss_history.append(loss)

        # ---------- PLOT ----------
        fig, ax = plt.subplots()
        ax.scatter(X, y, color="blue", alpha=0.7)

        x_line = np.linspace(x_min, x_max, 200)
        y_line = m * x_line + b

        ax.plot(x_line, y_line, color="red", linewidth=3)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.set_title(f"{gd_type} Gradient Descent – Step {step}")
        ax.set_xlabel("X")
        ax.set_ylabel("y")

        plot_area.pyplot(fig)

        time.sleep(0.25)

