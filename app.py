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


# 각 페이지에 대한 내용 표시
# 사이드바에 링크 추가
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["How to use?", "1.ㅤLog preprocessing", "2.ㅤFeature extract", "3.ㅤVisualization"])

# 각 페이지에 대한 내용 표시
if page == "How to use?":
    st.title("Instruction")
    st.markdown("#### 1. Log preprocessing")
    st.write("->  로그파일을 업로드하고 전처리가 되면, 전처리된 파일을 다운로드 해주세요.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 2. Feature extract")
    st.markdown("->  전처리된 로그파일을 업로드하고 피처 추출이 되면, 피처 추출된 파일을 다운로드 해주세요.")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 3. Visualization")
    st.markdown("->  피처 추출된 파일을 업로드하면, K-means,DBSCAN를 바탕으로 한 시각화를 할 수 있습니다.")
    
    # st.write("")
    # st.write("")
    # st.write("")
    # st.write("")
    # st.write("")
    # st.write("")
    # st.write("")
    # st.write("")
    # st.write("")
    # st.write("")
    
elif page == "1.ㅤLog preprocessing":
    st.title('로그 데이터 처리 앱')

    # 파일 업로드
    uploaded_file = st.file_uploader("CSV 파일 선택(1)", type="csv")

    if uploaded_file is not None:
        # CSV 파일 읽기
        log_df = pd.read_csv(uploaded_file)

        # 로그 데이터 처리
        processed_log_df = process_log_data(log_df)

        # 처리된 데이터 표시
        st.write("처리된 로그 데이터:")
        st.write(processed_log_df)

        # 처리된 데이터를 새로운 CSV 파일로 저장
        processed_file_path = 'processed_file.csv'
        processed_log_df.to_csv(processed_file_path, index=False)

        # 처리된 파일을 다운로드할 수 있는 링크 제공
        st.markdown(f"처리된 데이터 다운로드: [처리된 파일]({processed_file_path})")
elif page == "2.ㅤFeature extract":
    st.title('피처 추출 앱')
    
    # 파일 업로드
    uploaded_csvfile = st.file_uploader("CSV 파일 선택(2)", type="csv")

    if uploaded_csvfile is not None:
        # CSV 파일 읽기
        df_entity = pd.read_csv(uploaded_csvfile)

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

        # CSV 파일 다운로드 버튼 생성 (중복 제거된 데이터)
        csv_file_no_duplicates = df_entity_processed_no_duplicates.to_csv(index=False).encode()
        b64_no_duplicates = base64.b64encode(csv_file_no_duplicates).decode()
        st.button("Download CSV 파일 (중복 제거)", on_click=lambda: st.markdown(f'<a href="data:file/csv;base64,{b64_no_duplicates}" download="preprocessed_data_no_duplicates.csv">Download CSV 파일 (중복 제거)</a>', unsafe_allow_html=True))

elif page == "3.ㅤVisualization":
    st.title('Entity 클러스터링 및 시각화')

    # 파일 업로드
    uploaded_csvfile = st.file_uploader("CSV 파일 선택(3)", type="csv")

    if uploaded_csvfile is not None:
        # 업로드된 파일 읽기
        df_entity = pd.read_csv(uploaded_csvfile, index_col='entity')

        # Feature Scaling
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

        st.title('colab 코드 띄우기')
        st.write(df_entity['cluster_kmeans'].value_counts())
        st.write(df_entity[df_entity['cluster_kmeans']==0].index)
        
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



   
