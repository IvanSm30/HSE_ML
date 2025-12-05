import streamlit as st
import pickle


@st.cache_resource
def load_pipeline():
    with open("files/full_model_pipeline.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model():
    with open("files/linear_model_results.pkl", "rb") as f:
        return pickle.load(f)
