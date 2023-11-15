import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from process_log import process_log_data 
from feature_extraction import feature_extract
from feature_extraction import feature_extract2


# 각 페이지에 대한 내용 표시
# 사이드바에 링크 추가
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Contact"])

# 각 페이지에 대한 내용 표시
if page == "Home":
    st.title("Home Page")
    st.write("Welcome to the Home Page.")
elif page == "About":
    st.title("About Page")
    st.write("This is the About Page.")
elif page == "Contact":
    st.title("Contact Page")
    st.write("You can contact us here.")

   
