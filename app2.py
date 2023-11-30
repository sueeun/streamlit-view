import streamlit as st
import pandas as pd
import numpy as np
import base64
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from process_log import process_log_data 
from feature_extraction import feature_extract

# 사이드바
st.sidebar.title("ML Dashboard")

# 사이드바에 링크 추가
page = st.sidebar.radio("Go to", ["How to use?", "1.ㅤLog preprocessing", "2.ㅤFeature extract", "3.ㅤVisualization"])

# 각 페이지에 대한 내용 표시
if page == "How to use?":
    st.title("Instruction")
    st.markdown("***")
    st.markdown("##### 1. Log preprocessing")
    st.markdown("->ㅤ로그파일을 업로드하고 전처리가 되면, 전처리된 파일을 다운로드 해주세요.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 2. Feature extract")
    st.markdown("->ㅤ전처리된 로그파일을 업로드하고 피처 추출이 되면, 피처 추출된 파일을 다운로드 해주세요.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 3. Visualization")
    st.markdown("->ㅤ피처 추출된 파일을 업로드하면, K-means, DBSCAN를 바탕으로 한 시각화를 할 수 있습니다.")
    
elif page == "1.ㅤLog preprocessing":
    st.title('로그 데이터 처리 앱')
    st.markdown("***")
    
    # 파일 업로드
    log_file = st.file_uploader("CSV 파일 선택(1)", type="csv")

    if log_file is not None:
        # CSV 파일 읽기
        log_df = pd.read_csv(log_file)

        # 로그 데이터 처리
        processed_log_df = process_log_data(log_df)

        # 처리된 데이터 표시
        st.write("처리된 로그 데이터:")
        st.write(processed_log_df)

        # 처리된 데이터를 새로운 CSV 파일로 저장
        processed_file_path = 'processed_file.csv'
        processed_log_df.to_csv(processed_file_path, index=False)

elif page == "2.ㅤFeature extract":
    st.title('피처 추출 앱')
    st.markdown("***")
    
    # 파일 업로드
    processed_file = st.file_uploader("CSV 파일 선택(2)", type="csv")

    if processed_file is not None:
        # CSV 파일 읽기
        df_entity = pd.read_csv(processed_file)

        # Feature Extraction
        df_entity_processed = feature_extract(df_entity)

        # host 컬럼을 entity로 변경
        df_entity_processed = df_entity_processed.rename(columns={'Host': 'entity'})

        # 불필요한 컬럼 제거
        columns_to_drop = ['Unnamed: 0', 'Timestamp', 'Method', 'Protocol', 'Status', 'Referer', 'Path', 'UA', 'Payload', 'Bytes']
        df_entity_processed = df_entity_processed.drop(columns=columns_to_drop, errors='ignore')

        # 'entity' 컬럼을 맨 앞으로 이동
        columns_order = ['entity'] + [col for col in df_entity_processed.columns if col != 'entity']
        df_entity_processed = df_entity_processed[columns_order]

        # 중복된 행 제거
        df_entity_processed_no_duplicates = df_entity_processed.drop_duplicates()

        # 전처리된 데이터 출력 (중복 제거)
        st.write("전처리된 데이터 (중복 제거):")
        st.write(df_entity_processed_no_duplicates)
elif page == "3.ㅤVisualization":
    st.title('Entity 클러스터링 및 시각화')
    st.markdown("***")
    
    # 파일 업로드
    log_file = st.file_uploader("CSV 파일 선택(1)", type="csv")
    feature_file = st.file_uploader("CSV 파일 선택(3)", type="csv")

    if log_file is not None:
        log_df = pd.read_csv(log_file)
    
    if feature_file is not None:
        # 업로드된 파일 읽기
        df_entity = pd.read_csv(feature_file, index_col='entity')

        # Feature Scaling
        columns_to_scale = ['method_cnt', 'status_cnt', 'ua_cnt', 'bytes_avg', 'bytes_std']
        scaler = preprocessing.MinMaxScaler()
        scaler = scaler.fit(df_entity[columns_to_scale])
        df_entity[columns_to_scale] = scaler.transform(df_entity[columns_to_scale])
        
        cols_to_train = ['method_cnt','method_post','protocol_1_0','status_major','status_404','status_499','status_cnt','path_same','path_xmlrpc','ua_cnt','has_payload','bytes_avg','bytes_std']
        
        # -- 모델링 --
        # Kmeans
        kmeans = KMeans (n_clusters=2, random_state=42)
        kmeans.fit(df_entity[cols_to_train])
        df_entity['cluster_kmeans'] = kmeans.predict(df_entity[cols_to_train])

        # DBSCAN
        dbscan = DBSCAN(eps=0.5,min_samples=2)
        dbscan.fit(df_entity[cols_to_train])
        df_entity['cluster_dbscan'] = dbscan.fit_predict(df_entity[cols_to_train])

        # 데이터프레임 출력
        st.write(df_entity)
        
        # -- 막대그래프 --
        st.markdown("### 1. 막대그래프")
        
        # Kmeans
        kmeans_value_counts = df_entity['cluster_kmeans'].value_counts()
        x = np.arange(2)
        
        result = kmeans_value_counts.index.values
        count = kmeans_value_counts.values
        
        fig_kmeans_bar = plt.figure(figsize=(10, 6))
        plt.bar(x, count)
        plt.xticks(x, result)
        
        for i, value in enumerate(result):
           plt.text(x[i], count[i], count[i], ha='center', va='bottom')
        
        st.markdown("#### Kmeans")
        st.pyplot(fig_kmeans_bar)
        st.markdown(
          """
          0이 이상탐지된 아이피, 1이 정상아이피이다.
          """
        )
        
        # DBSCAN
        dbscan_value_counts = df_entity['cluster_dbscan'].value_counts()
        x = np.arange(2)

        dbscan_index = dbscan_value_counts.index.values
        indices = np.where(dbscan_index == 0)
        
        zero_index = int(list(indices)[0])
        
        result = [dbscan_value_counts.index.values[zero_index], sorted(dbscan_value_counts.index.values[zero_index+1:].tolist()+dbscan_value_counts.index.values[:zero_index].tolist())]
        count = [dbscan_value_counts.values[zero_index], dbscan_value_counts.values.sum()-dbscan_value_counts.values[zero_index]]
        
        fig_dbscan_bar = plt.figure(figsize=(10, 6))
        plt.bar(x, count)
        plt.xticks(x, result)
        
        for i, value in enumerate(result):
           plt.text(x[i], count[i], count[i], ha='center', va='bottom')
          
        st.markdown("#### DBSCAN")
        st.pyplot(fig_dbscan_bar)
        st.markdown(
          """
          0이 정상아이피, 이외는 이상탐지된 아이피이다.
          """
        )
        
        anomalyDetection_kmeans = df_entity[df_entity['cluster_kmeans'] == 0].index
        anomalyDetection_dbscan = df_entity[df_entity['cluster_dbscan'] != 0].index
        
        # 아이피, 로그 조회
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Kmeans와 DBSCAN에서 이상탐지된 아이피 조회")
        st.write(anomalyDetection_kmeans, anomalyDetection_dbscan)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("#### kmeans, dbscan에 이상탐지된 아이피 검색")
        search_entity = st.text_input("검색할 ip 입력:", key="input1")
        
        if search_entity:
          st.write("검색 결과:")
          if search_entity in anomalyDetection_kmeans and search_entity in anomalyDetection_dbscan:
              st.write(f"{search_entity}은(는) KMeans, DBSCAN 두 모델로부터 이상탐지 되었습니다.")
          elif search_entity in anomalyDetection_kmeans:
              st.write(f"{search_entity}은(는) KMeans 모델로부터 이상탐지 되었습니다.")
          elif search_entity in anomalyDetection_dbscan:
              st.write(f"{search_entity}은(는) DBSCAN 모델로부터 이상탐지 되었습니다.")
          else:
              st.write(f"{search_entity}은(는) KMeans, DBSCAN 두 모델로부터 이상탐지 되지 않았습니다.")
        
        st.markdown("#### 아이피의 로그 검색")
        searched_ip = st.text_input("검색할 ip 입력:", key="input2")
        
        if searched_ip:
          st.write("검색 결과:")
          st.write(log_df[log_df['message'].str.contains(searched_ip)].drop('timestamp', axis=1)) 
          
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        

        # -- PCA --
        st.markdown('### 2. PCA 그래프')
        
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
        plt.colorbar(label='cluster')

        st.markdown("#### Kmeans")
        st.pyplot(fig_kmeans)

        # 2D PCA 결과를 시각화
        fig_dbscan = plt.figure(figsize=(10, 6))
        plt.scatter(df_entity['pca_1'], df_entity['pca_2'], c=df_entity['cluster_dbscan'], cmap='viridis', s=60)
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.colorbar(label='cluster')

        st.markdown("#### DBSCAN")
        st.pyplot(fig_dbscan)



   
