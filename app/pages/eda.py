import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

palette = (
    sns.color_palette("tab10") + sns.color_palette("Set3") + sns.color_palette("Dark2")
)

df_train = pd.read_csv("./files/df_train.csv")
df_test = pd.read_csv("./files/df_test.csv")

X_train = df_train.drop(["selling_price", "name"], axis=1)
y_train = df_train["selling_price"]
y_train_log = np.log(y_train)

X_test = df_test.drop(["selling_price", "name"], axis=1)
y_test = df_test["selling_price"]
y_test_log = np.log(y_test)

numeric_cols = X_train.select_dtypes(include=["number"])
cat_cols = list(X_train.select_dtypes(include=["object"]).columns)
train_corr_matrix = numeric_cols.corr()

st.header("Visualization")

sample = st.selectbox(
    "Choice sample",
    ["train", "test"],
    index=None,
    placeholder="Select sample type",
)

option = None
if sample:
    option = st.selectbox(
        "Choice graphic",
        ["Pairwise_distributions", "Corr", "Categorical_attributes_from_prices"],
        index=None,
        placeholder="Select one type of the graphic",
    )

X, y, y_log, numeric_data = None, None, None, None
if sample == "train":
    X = X_train
    y = y_train
    y_log = y_train_log
    numeric_data = pd.concat(
        [X_train.select_dtypes(include=["number"]), y_train], axis=1
    )
elif sample == "test":
    X = X_test
    y = y_test
    y_log = y_test_log
    numeric_data = pd.concat([X_test.select_dtypes(include=["number"]), y_test], axis=1)

if option is None:
    if sample:
        st.info("Please select a graphic type.")
    else:
        st.info("Please select a sample (train/test) first.")
else:
    match option:
        case "Pairwise_distributions":
            if numeric_data is not None:
                fig_pairplot = sns.pairplot(numeric_data)
                st.pyplot(fig_pairplot)
            else:
                st.error("No numeric data available for the selected sample.")

        case "Corr":
            if sample == "train":
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    train_corr_matrix, annot=True, fmt=".3f", vmin=-1, vmax=1, ax=ax
                )
                st.pyplot(fig)
            else:
                st.info("Correlation matrix is only available for the training sample.")

        case "Categorical_attributes_from_prices":
            if X is None or y is None:
                st.error("Data not available for categorical plots.")
            else:
                type_y = st.selectbox(
                    "Choice y",
                    ["normal", "log"],
                    index=None,
                    placeholder="Select y type",
                )
                if type_y is None:
                    st.info("Please select y type (normal or log).")
                else:
                    y_to_use = y_log if type_y == "log" else y

                    n_cats = len(cat_cols)
                    if n_cats == 0:
                        st.warning("No categorical columns found.")
                    else:

                        fig, axes = plt.subplots(1, n_cats, figsize=(5 * n_cats, 6))
                        if n_cats == 1:
                            axes = [axes]
                            

                        for i, cat in enumerate(cat_cols):
                            if cat in X.columns:
                                n_colors = X[cat].nunique()
                                current_palette = sns.color_palette("husl", n_colors)
                                sns.boxplot(
                                    data=X,
                                    x=cat,
                                    y=y_to_use,
                                    ax=axes[i],
                                    palette=current_palette,
                                )
                                axes[i].tick_params(axis="x", rotation=45)
                                axes[i].set_title(cat)
                            else:
                                axes[i].set_visible(False)
                                st.warning(
                                    f"Column '{cat}' not found in selected sample."
                                )

                        plt.tight_layout()
                        st.pyplot(fig)
