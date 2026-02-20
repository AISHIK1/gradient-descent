import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_regression

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(layout="wide")
st.title("Gradient Descent Line Animation (Epoch by Epoch)")

st.markdown("""
This demo shows **how the regression line evolves after every update of**
**m (slope)** and **b (intercept)** using Gradient Descent.
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
m = st.sidebar.slider("Initial m", -50.0, 50.0, -30.0)
b = st.sidebar.slider("Initial b", -200.0, 200.0, -100.0)

run = st.sidebar.button("▶ Run Gradient Descent")

# ---------------------------------
# Dataset (EXACTLY as requested)
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

        # -----------------------------
        # Gradient update
        # -----------------------------
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
                Xb = Xs[i:i+batch_size]
                yb = ys[i:i+batch_size]
                y_hat = m * Xb + b
                dm = (-2/len(Xb)) * np.sum(Xb * (yb - y_hat))
                db = (-2/len(Xb)) * np.sum(yb - y_hat)
                m -= lr * dm
                b -= lr * db

        # -----------------------------
        # Loss
        # -----------------------------
        loss = mse(y, m * X + b)
        loss_history.append(loss)

        # -----------------------------
        # Plot regression line (KEY PART)
        # -----------------------------
        fig, ax = plt.subplots()

        ax.scatter(X, y, color="blue", label="Data", alpha=0.7)

        x_line = np.linspace(X.min() - 1, X.max() + 1, 100)
        y_line = m * x_line + b

        ax.plot(
            x_line, y_line,
            color="red",
            linewidth=2,
            label=f"Epoch {epoch}"
        )

        ax.set_title(f"Line after Epoch {epoch}")
        ax.set_xlabel("X")
        ax.set_ylabel("y")
        ax.legend()

        plot_area.pyplot(fig)

        # -----------------------------
        # Loss curve
        # -----------------------------
        fig2, ax2 = plt.subplots()
        ax2.plot(loss_history, color="purple")
        ax2.set_title("Loss vs Epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MSE")
        loss_area.pyplot(fig2)

        time.sleep(0.25)
