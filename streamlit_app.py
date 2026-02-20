import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Gradient Descent Line Evolution",
    layout="wide"
)

st.title("📉 Gradient Descent: Line Evolution Demo")
st.markdown(
    """
    This demo shows **how a regression line evolves from an initial guess (m₀, b₀)**
    into the optimal regression line using Gradient Descent.
    """
)

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("⚙️ Controls")

gd_type = st.sidebar.selectbox(
    "Gradient Descent Type",
    ["Batch", "Mini-Batch", "Stochastic"]
)

learning_rate = st.sidebar.slider("Learning Rate (η)", 0.001, 0.1, 0.01)
epochs = st.sidebar.slider("Epochs", 10, 200, 60)

batch_size = st.sidebar.slider(
    "Mini-Batch Size",
    2, 20, 5,
    disabled=(gd_type != "Mini-Batch")
)

# Initial parameters
st.sidebar.subheader("Initial Line Parameters")
m = st.sidebar.slider("Initial slope (m₀)", -10.0, 10.0, -5.0)
b = st.sidebar.slider("Initial intercept (b₀)", -10.0, 10.0, 8.0)

run = st.sidebar.button("▶ Start Animation")

# -------------------------------
# Generate dataset
# -------------------------------
np.random.seed(42)
X = np.linspace(0, 10, 50)
true_m = 2.5
true_b = 3.0
y = true_m * X + true_b + np.random.randn(50)

n = len(X)

# -------------------------------
# Loss function
# -------------------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# -------------------------------
# Plot placeholders
# -------------------------------
col1, col2 = st.columns(2)

line_plot = col1.empty()
param_plot = col2.empty()
loss_plot = st.empty()

m_history = []
b_history = []
loss_history = []

# -------------------------------
# Gradient Descent Animation
# -------------------------------
if run:
    for epoch in range(epochs):

        # ---------------------------
        # Gradient updates
        # ---------------------------
        if gd_type == "Batch":
            y_pred = m * X + b
            dm = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)
            m -= learning_rate * dm
            b -= learning_rate * db

        elif gd_type == "Stochastic":
            for i in range(n):
                xi, yi = X[i], y[i]
                y_pred = m * xi + b
                dm = -2 * xi * (yi - y_pred)
                db = -2 * (yi - y_pred)
                m -= learning_rate * dm
                b -= learning_rate * db

        elif gd_type == "Mini-Batch":
            indices = np.random.permutation(n)
            Xs, ys = X[indices], y[indices]
            for i in range(0, n, batch_size):
                Xb = Xs[i:i+batch_size]
                yb = ys[i:i+batch_size]
                y_pred = m * Xb + b
                dm = (-2/len(Xb)) * np.sum(Xb * (yb - y_pred))
                db = (-2/len(Xb)) * np.sum(yb - y_pred)
                m -= learning_rate * dm
                b -= learning_rate * db

        # ---------------------------
        # Store history
        # ---------------------------
        m_history.append(m)
        b_history.append(b)
        loss_history.append(mse(y, m * X + b))

        # ---------------------------
        # Plot line evolution
        # ---------------------------
        fig1, ax1 = plt.subplots()
        ax1.scatter(X, y, label="Data")
        ax1.plot(X, m * X + b, color="red", label="Current line")
        ax1.plot(X, true_m * X + true_b, "--", color="green", label="True line")
        ax1.set_title(f"Epoch {epoch + 1}")
        ax1.legend()
        line_plot.pyplot(fig1, width="stretch")

        # ---------------------------
        # Plot m & b convergence
        # ---------------------------
        fig2, ax2 = plt.subplots()
        ax2.plot(m_history, label="m (slope)")
        ax2.plot(b_history, label="b (intercept)")
        ax2.axhline(true_m, linestyle="--", color="gray")
        ax2.axhline(true_b, linestyle=":", color="gray")
        ax2.set_title("Parameter Convergence")
        ax2.legend()
        param_plot.pyplot(fig2, width="stretch")

        # ---------------------------
        # Plot loss curve
        # ---------------------------
        fig3, ax3 = plt.subplots()
        ax3.plot(loss_history, color="purple")
        ax3.set_title("Loss (MSE)")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        loss_plot.pyplot(fig3, width="stretch")

        time.sleep(0.1)
