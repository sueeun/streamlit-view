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
from matplotlib import font_manager, rc

font_path = "NanumGothic.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

def main():
   df_entity = pd.read_csv('pj_processed.csv', index_col='entity')
   # st.write(df_entity)
   columns_to_scale = ['method_cnt', 'status_cnt', 'ua_cnt', 'bytes_avg', 'bytes_std']
   scaler = preprocessing.MinMaxScaler()
   scaler = scaler.fit(df_entity[columns_to_scale])
   df_entity[columns_to_scale] = scaler.transform(df_entity[columns_to_scale])

   cols_to_train = ['method_cnt','method_post','protocol_1_0','status_major','status_404','status_499','status_cnt','path_same','path_xmlrpc','ua_cnt','has_payload','bytes_avg','bytes_std']

   # Kmeans
   kmeans = KMeans (n_clusters=2, random_state=42)
   kmeans.fit(df_entity[cols_to_train])
   df_entity['cluster_kmeans'] = kmeans.predict(df_entity[cols_to_train])
   
   # DBSCAN
   dbscan = DBSCAN(eps=0.5,min_samples=2)
   dbscan.fit(df_entity[cols_to_train])
   df_entity['cluster_dbscan'] = dbscan.fit_predict(df_entity[cols_to_train])

   # st.write(df_entity

   st.title('colab 코드 띄우기')
   st.write(df_entity['cluster_kmeans'].value_counts())
   st.write(df_entity[df_entity['cluster_kmeans']==0].index)

   st.title('PCA 그래프')

   # PCA를 사용하여 데이터의 차원을 2로 축소
   pca = PCA(n_components=2)
   pca_result = pca.fit_transform(df_entity[cols_to_train])
   
   # PCA 결과를 데이터프레임에 추가
   df_entity['pca_1'] = pca_result[:, 0]
   df_entity['pca_2'] = pca_result[:, 1]

   # 2D PCA 결과를 시각화
   fig_kmeans = plt.figure(figsize=(10, 6))
   plt.scatter(df_entity['pca_1'], df_entity['pca_2'], c=df_entity['cluster_kmeans'], cmap='viridis', s=60)
   plt.xlabel("PCA 1")
   plt.ylabel("PCA 2")
   plt.title("KMeans 클러스터링된 Entity 시각화 (PCA 결과)")
   plt.colorbar(label='클러스터')
   
   st.pyplot(fig_kmeans)

   # 2D PCA 결과를 시각화
   fig_dbscan = plt.figure(figsize=(10, 6))
   plt.scatter(df_entity['pca_1'], df_entity['pca_2'], c=df_entity['cluster_dbscan'], cmap='viridis', s=60)
   plt.xlabel("PCA 1")
   plt.ylabel("PCA 2")
   plt.title("DBSCAN 클러스터링된 Entity 시각화 (PCA 결과)")
   plt.colorbar(label='클러스터')
   
   st.pyplot(fig_dbscan)
   


   # -- 막대 그래프 --
   st.title('막대그래프')
   
   # Kmeans
   kmans_value_counts = df_entity['cluster_kmeans'].value_counts()
   x = np.arange(2)
   
   result = kmans_value_counts.index.values
   count = kmans_value_counts.values
   
   plt.bar(x, count)
   plt.xticks(x, result)
   
   for i, value in enumerate(result):
      plt.text(x[i], count[i], count[i], ha='center', va='bottom')

   plt.show()

   # # DBSCAN
   # dbscan_value_counts = df_entity['cluster_dbscan'].value_counts()
   # x = np.arange(2)

   # result = [dbscan_value_counts.index.values[0],dbscan_value_counts.index.values[1:]]
   # count = [dbscan_value_counts.values[0],dbscan_value_counts.values[1:].sum()]
   
   # plt.bar(x, count)
   # plt.xticks(x, result)
   
   # for i, value in enumerate(result):
   #   plt.text(x[i], count[i], count[i], ha='center', va='bottom')
   
   # plt.show()

   # # 아이피 띄우기
   # st.title('이상탐지된 아이피')
   # df_entity[df_entity['cluster_kmeans']!=1].index
   # df_entity[df_entity['cluster_dbscan']!=0].index
   
   
if __name__ == '__main__':
    main()
