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



def main():
   df_entity = pd.read_csv('pj_processed.csv', index_col='entity')
   st.write(df_entity)




if __name__ == '__main__':
    main()
