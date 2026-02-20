import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_regression

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(layout="wide")
st.title("Gradient Descent: Line Movement + Contour Plot")

st.markdown("""
This visualization shows:

• **Left:** Regression line evolving step-by-step  
• **Right:** Contour plot of loss J(m, b)  
• The red dot on the contour shows the **current (m, b)**  

Each animation step corresponds to **one gradient update**.
""")

# -------------------------------------------------
# Sidebar controls
# -------------------------------------------------
st.sidebar.header("Controls")

gd_type = st.sidebar.selectbox(
    "Gradient Descent Type",
    ["Batch", "Stochastic", "Mini-Batch"]
)

lr = st.sidebar.slider("Learning Rate (η)", 0.0001, 0.05, 0.01)
steps = st.sidebar.slider("Steps (updates)", 5, 200, 50)

batch_size = st.sidebar.slider(
    "Mini-Batch Size",
    2, 20, 10,
    disabled=(gd_type != "Mini-Batch")
)

st.sidebar.subheader("Initial Parameters")
m = st.sidebar.slider("Initial m", -50.0, 50.0, -30.0)
b = st.sidebar.slider("Initial b", -200.0, 200.0, -100.0)

run = st.sidebar.button("▶ Run Gradient Descent")

# -------------------------------------------------
# Dataset (fixed)
# -------------------------------------------------
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

# -------------------------------------------------
# Fixed axis limits (CRITICAL)
# -------------------------------------------------
x_min, x_max = X.min() - 1, X.max() + 1
y_min, y_max = y.min() - 50, y.max() + 50

# -------------------------------------------------
# Loss function
# -------------------------------------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# -------------------------------------------------
# Precompute contour surface (ONCE)
# -------------------------------------------------
m_vals = np.linspace(-50, 50, 100)
b_vals = np.linspace(-200, 200, 100)

M, B = np.meshgrid(m_vals, b_vals)
Z = np.zeros_like(M)

for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        Z[i, j] = mse(y, M[i, j] * X + B[i, j])

# -------------------------------------------------
# Layout
# -------------------------------------------------
col1, col2 = st.columns(2)
line_plot = col1.empty()
contour_plot = col2.empty()

# -------------------------------------------------
# Gradient Descent Animation
# -------------------------------------------------
if run:

    for step in range(steps):

        # ---------- BATCH ----------
        if gd_type == "Batch":
            y_hat = m * X + b
            dm = (-2/n) * np.sum(X * (y - y_hat))
            db = (-2/n) * np.sum(y - y_hat)
            m -= lr * dm
            b -= lr * db

        # ---------- STOCHASTIC ----------
        elif gd_type == "Stochastic":
            i = np.random.randint(n)
            xi, yi = X[i], y[i]
            y_hat = m * xi + b
            dm = -2 * xi * (yi - y_hat)
            db = -2 * (yi - y_hat)
            m -= lr * dm
            b -= lr * db

        # ---------- MINI-BATCH ----------
        elif gd_type == "Mini-Batch":
            idx = np.random.choice(n, batch_size, replace=False)
            Xb, yb = X[idx], y[idx]
            y_hat = m * Xb + b
            dm = (-2/len(Xb)) * np.sum(Xb * (yb - y_hat))
            db = (-2/len(Xb)) * np.sum(yb - y_hat)
            m -= lr * dm
            b -= lr * db

        # -------------------------------------------------
        # LEFT: Regression line
        # -------------------------------------------------
        fig1, ax1 = plt.subplots()
        ax1.scatter(X, y, color="blue", alpha=0.6)

        x_line = np.linspace(x_min, x_max, 200)
        y_line = m * x_line + b
        ax1.plot(x_line, y_line, color="red", linewidth=3)

        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_title(f"{gd_type} – Line (Step {step+1})")
        ax1.set_xlabel("X")
        ax1.set_ylabel("y")

        line_plot.pyplot(fig1)

        # -------------------------------------------------
        # RIGHT: Contour plot
        # -------------------------------------------------
        fig2, ax2 = plt.subplots()
        ax2.contour(M, B, Z, levels=30, cmap="viridis")
        ax2.scatter(m, b, color="red", s=60)

        ax2.set_title("Loss Contour J(m, b)")
        ax2.set_xlabel("m (slope)")
        ax2.set_ylabel("b (intercept)")

        contour_plot.pyplot(fig2)

        time.sleep(0.25)
