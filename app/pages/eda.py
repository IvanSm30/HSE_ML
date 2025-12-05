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
        ["Pairwise_distributions", "Corr", "Boxplot"],
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

        case "Boxplot":
            if X is None or y is None:
                st.error("Data not available for boxplot.")
                pass

            cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)

            if len(cat_cols) == 0:
                st.warning("No categorical columns found in the selected sample.")
                pass

            type_y = st.radio(
                "Target scale", options=["normal", "log"], index=0, horizontal=True
            )
            y_to_use = np.log(y) if type_y == "log" else y

            selected_cats = st.multiselect(
                "Select categorical variables to plot",
                options=cat_cols,
                default=cat_cols[:4]
                if len(cat_cols) >= 4
                else cat_cols,  # по умолчанию первые 4
            )

            if not selected_cats:
                st.info("Please select at least one categorical variable.")
            else:
                n_plots = len(selected_cats)
                cols = min(n_plots, 4)
                rows = (n_plots + cols - 1) // cols

                fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
                if n_plots == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()

                for i, cat_col in enumerate(selected_cats):
                    ax = axes[i]
                    unique_vals = X[cat_col].nunique()
                    palette_i = sns.color_palette("husl", unique_vals)

                    sns.boxplot(data=X, x=cat_col, y=y_to_use, ax=ax, palette=palette_i)
                    ax.set_title(cat_col)
                    ax.tick_params(axis="x", rotation=45)
                    ax.set_ylabel(
                        "selling_price" + (" (log)" if type_y == "log" else "")
                    )

                for j in range(n_plots, len(axes)):
                    axes[j].set_visible(False)

                plt.tight_layout()
                st.pyplot(fig)
            if X is None or y is None:
                st.error("Data not available for boxplot.")
            else:
                numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
                available_vars = ["selling_price"] + numeric_columns

                selected_vars = st.multiselect(
                    "Select numeric variables to plot (including selling_price)",
                    options=available_vars,
                    default=["selling_price"],
                )

                if not selected_vars:
                    st.info("Please select at least one variable.")
                else:
                    use_log = False
                    if "selling_price" in selected_vars:
                        type_y = st.radio(
                            "Target scale",
                            options=["normal", "log"],
                            index=0,
                            horizontal=True,
                            key="type_y"
                        )
                        use_log = type_y == "log"

                    plot_data = X.copy()
                    if "selling_price" in selected_vars:
                        plot_data["selling_price"] = np.log(y) if use_log else y

                    selected_vars = [
                        col for col in selected_vars if col in plot_data.columns
                    ]

                    if not selected_vars:
                        st.warning("No valid columns selected.")
                    else:
                        n_vars = len(selected_vars)
                        cols = min(n_vars, 3)
                        rows = (n_vars + cols - 1) // cols

                        fig, axes = plt.subplots(
                            rows, cols, figsize=(5 * cols, 4 * rows)
                        )
                        if n_vars == 1:
                            axes = [axes]
                        else:
                            axes = axes.flatten()

                        for i, var in enumerate(selected_vars):
                            sns.boxplot(
                                y=plot_data[var],
                                ax=axes[i],
                                width=0.5,
                                color=palette[i % len(palette)],
                            )
                            axes[i].set_title(f"Boxplot: {var}")
                            axes[i].set_ylabel(var)

                        for j in range(n_vars, len(axes)):
                            axes[j].set_visible(False)

                        plt.tight_layout()
                        st.pyplot(fig)
