import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Gradient Descent Animation",
    layout="wide"
)

st.title("📉 Gradient Descent Animation (Batch | Mini-Batch | Stochastic)")
st.markdown("Visualizing **m**, **b**, and **loss** during training")

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("⚙️ Controls")

gd_type = st.sidebar.selectbox(
    "Select Gradient Descent Type",
    ["Batch", "Mini-Batch", "Stochastic"]
)

learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01)
epochs = st.sidebar.slider("Epochs", 10, 200, 50)
batch_size = st.sidebar.slider("Mini-Batch Size", 2, 20, 5)

animate = st.sidebar.button("▶ Run Animation")

# -------------------------------
# Generate dataset
# -------------------------------
np.random.seed(42)
X = np.linspace(0, 10, 50)
y = 2.5 * X + 3 + np.random.randn(50)

# -------------------------------
# Initialize parameters
# -------------------------------
m = 0.0
b = 0.0
n = len(X)

m_history = []
b_history = []
loss_history = []

# -------------------------------
# Loss function
# -------------------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# -------------------------------
# Plot containers
# -------------------------------
col1, col2 = st.columns(2)

line_plot = col1.empty()
param_plot = col2.empty()
loss_plot = st.empty()

# -------------------------------
# Gradient Descent Logic
# -------------------------------
if animate:
    for epoch in range(epochs):

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
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                y_pred = m * X_batch + b
                dm = (-2/len(X_batch)) * np.sum(X_batch * (y_batch - y_pred))
                db = (-2/len(X_batch)) * np.sum(y_batch - y_pred)

                m -= learning_rate * dm
                b -= learning_rate * db

        # -------------------------------
        # Store history
        # -------------------------------
        m_history.append(m)
        b_history.append(b)
        loss_history.append(mse(y, m * X + b))

        # -------------------------------
        # Plot regression line
        # -------------------------------
        fig1, ax1 = plt.subplots()
        ax1.scatter(X, y, label="Data")
        ax1.plot(X, m * X + b, color="red", label="Model")
        ax1.set_title(f"Epoch {epoch+1}")
        ax1.legend()
        line_plot.pyplot(fig1, width="stretch")

        # -------------------------------
        # Plot m and b
        # -------------------------------
        fig2, ax2 = plt.subplots()
        ax2.plot(m_history, label="m (slope)")
        ax2.plot(b_history, label="b (intercept)")
        ax2.set_title("Parameter Convergence")
        ax2.legend()
        param_plot.pyplot(fig2, width="stretch")

        # -------------------------------
        # Plot loss
        # -------------------------------
        fig3, ax3 = plt.subplots()
        ax3.plot(loss_history, color="purple")
        ax3.set_title("Loss (MSE)")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        loss_plot.pyplot(fig3, width="stretch")

        time.sleep(0.1)
