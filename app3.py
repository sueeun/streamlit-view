import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from process_log import process_log_data 
from feature_extraction import feature_extract
from feature_extraction import feature_extract2



def main():
   df_entity = pd.read_csv('pj_processed.csv', index_col='entity')
   # st.write(df_entity)
   columns_to_scale = ['method_cnt', 'status_cnt', 'ua_cnt', 'bytes_avg', 'bytes_std']
   scaler = preprocessing.MinMaxScaler()
   scaler = scaler.fit(df_entity[columns_to_scale])
   df_entity[columns_to_scale] = scaler.transform(df_entity[columns_to_scale])

   cols_to_train = ['method_cnt','method_post','protocol_1_0','status_major','status_404','status_499','status_cnt','path_same','path_xmlrpc','ua_cnt','has_payload','bytes_avg','bytes_std']
   kmeans = KMeans (n_clusters=2, random_state=42)
   # 정상/ 비정상 클러스터로 나누어 보기
   kmeans.fit(df_entity[cols_to_train])

   df_entity['cluster_kmeans'] = kmeans.predict(df_entity[cols_to_train])
   st.write(df_entity)
   # st.write(df_entity['cluster_kmeans'].value_counts())


if __name__ == '__main__':
    main()
