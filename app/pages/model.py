import streamlit as st
import pickle
import matplotlib.pyplot as plt
import pandas as pd


st.header("Model")

@st.cache_resource
def load_model():
    with open("files/linear_model_results.pkl", "rb") as f:
        return pickle.load(f)
    
model_data = load_model()

if not model_data.get("feature_names") or not model_data.get("coefficients"):
    st.error("В сохранённой модели отсутствуют коэффициенты или названия признаков.")
    st.stop()

coef_df = pd.DataFrame(
    {
        "feature": model_data["feature_names"],
        "coefficient": model_data["coefficients"],
    }
)

coef_df = coef_df.reindex(
    coef_df["coefficient"].abs().sort_values(ascending=True).index
)

fig, ax = plt.subplots(figsize=(10, max(6, len(coef_df) * 0.4)))

# Цвета: красный — отрицательный, синий — положительный
colors = ["red" if x < 0 else "steelblue" for x in coef_df["coefficient"]]

# Горизонтальный bar-график через matplotlib
ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors)

# Вертикальная линия через 0
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

ax.set_title("Model Coefficients (Linear Regression)", fontsize=16)
ax.set_xlabel("Coefficient value")
ax.set_ylabel("Feature")

# Повернуть подписи, если нужно
# ax.tick_params(axis='y', labelsize=9)

plt.tight_layout()
st.pyplot(fig)
