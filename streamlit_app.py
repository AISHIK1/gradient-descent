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
if run:

    for epoch in range(epochs):

        # -------- Gradient update --------
        if gd_type == "Batch":
            y_hat = m * X + b
            dm = (-2/n) * np.sum(X * (y - y_hat))
            db = (-2/n) * np.sum(y - y_hat)
            m -= lr * dm
            b -= lr * db

        elif gd_type == "Stochastic":
            for i in range(n):
                xi, yi = X[i], y[i]
                y_hat = m * xi + b
                dm = -2 * xi * (yi - y_hat)
                db = -2 * (yi - y_hat)
                m -= lr * dm
                b -= lr * db

        elif gd_type == "Mini-Batch":
            indices = np.random.permutation(n)
            Xs, ys = X[indices], y[indices]
            for i in range(0, n, batch_size):
                Xb = Xs[i:i + batch_size]
                yb = ys[i:i + batch_size]
                y_hat = m * Xb + b
                dm = (-2 / len(Xb)) * np.sum(Xb * (yb - y_hat))
                db = (-2 / len(Xb)) * np.sum(yb - y_hat)
                m -= lr * dm
                b -= lr * db

        # -------- Loss --------
        loss = mse(y, m * X + b)
        loss_history.append(loss)

        # -------- Plot (POINTS DO NOT MOVE) --------
        fig, ax = plt.subplots()

        ax.scatter(X, y, color="blue", alpha=0.7, label="Data")

        x_line = np.linspace(x_min, x_max, 200)
        y_line = m * x_line + b

        ax.plot(
            x_line,
            y_line,
            color="red",
            linewidth=3,
            label=f"Epoch {epoch}"
        )

        # LOCKED AXES (KEY FIX)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.set_title(f"Regression Line – Epoch {epoch}")
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.legend()

        plot_area.pyplot(fig)

        # -------- Loss plot --------
        fig2, ax2 = plt.subplots()
        ax2.plot(loss_history, color="purple")
        ax2.set_title("Loss vs Epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MSE")
        loss_area.pyplot(fig2)

        time.sleep(0.25)
