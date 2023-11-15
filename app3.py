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

uploaded_file = None

# 사이드바에 링크 추가
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Contact"])

# 각 페이지에 대한 내용 표시
if page == "Home":
    st.title('로그 데이터 처리 앱')

    # 파일 업로드
    global uploaded_file = st.file_uploader("CSV 파일 선택", type="csv")

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
elif page == "About":
    uploaded_file.close()
    st.title('로그 데이터 처리 앱')

    # 파일 업로드
    global uploaded_file = st.file_uploader("CSV 파일 선택", type="csv")

    if uploaded_csvfile is not None:
        # CSV 파일 읽기
        df_entity = pd.read_csv(uploaded_csvfile)

        # Feature Extraction
        df_entity_processed = feature_extract(df_entity)

        # 전처리된 데이터 출력
        st.write("전처리된 데이터:")
        st.write(df_entity_processed)
elif page == "Contact":
    uploaded_file.close()
    
    st.title('로그 데이터 처리 앱')

    # 파일 업로드
    global uploaded_file = st.file_uploader("CSV 파일 선택", type="csv")

    if uploaded_csvfile is not None:
        # CSV 파일 읽기
        df_entity = pd.read_csv(uploaded_csvfile)

        # Feature Extraction 수행
        df_entity_processed = feature_extract(df_entity)

        # 전처리 결과 출력
        st.write("전처리된 데이터:")
        st.write(df_entity_processed)

        # 결과를 CSV로 저장
        save_button = st.button("결과 저장")
        if save_button:
            df_entity_processed.to_csv("preprocessed_data.csv", index=False)
            st.success("결과가 성공적으로 저장되었습니다.")
