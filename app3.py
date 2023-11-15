import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.decomposition import PCA
import joblib
from feature_extraction import feature_extract

def process_log_data(log_df):
    log_df.drop(columns='timestamp', inplace=True)
    log_df['Timestamp'] = log_df['message'].str.extract(r'(\d+/\w+/\d+\d+:\d+:\d+:\d+)')
    log_df['Timestamp'] = pd.to_datetime(log_df['Timestamp'], format='%d/%b/%Y:%H:%M:%S').dt.strftime('%Y-%m-%d %H:%M:%S')
    log_df['Host'] = log_df['message'].str.extract(r'(\d+.\d+.\d+.\d+)')
    log_df[['Method', 'Path']] = log_df['message'].str.extract(r'(HEAD|PUT|DELETE|CONNECT|OPTIONS|TRACE|PATCH|POST|GET)\s+(.?)\s+HTTP')
    log_df['Protocol'] = log_df['message'].str.extract(r'(HTTP/\d+.\d+)')
    log_df['Status'] = log_df['message'].str.extract(r'(\d+)\s+\d+')
    log_df['Bytes'] = log_df['message'].str.extract(r'\d+\s+(\d+)')
    log_df['UA'] = log_df['message'].str.extract(r'(Mozilla.+537.36)')
    selected_log_df = log_df[log_df['Method'].isna() & log_df['Protocol'].isna()]
    log_df['Payload'] = selected_log_df['message'].str.extract(r']{1}\s+"(.)" \d+')
    log_df['Referer'] = log_df['message'].str.extract(r'."(http[s]?://.?)"')
    log_df.drop(columns='message', inplace=True)
    log_df = log_df[['Timestamp','Method','Protocol','Status','Referer','Path','Host','UA','Payload','Bytes']]
    return log_df

def feature_extract(df):
    # 사용할 feature 목록
    cols_to_train = ['Method', 'Protocol', 'Status', 'Referer', 'Path', 'UA', 'Payload', 'Bytes']

    # 사용자 정의 함수: Method 관련 Feature
    def extract_method_features(group):
        method_cnt = group['Method'].nunique()
        method_post_percent = len(group[group['Method'] == 'POST']) / max(1, len(group))
        return method_cnt, method_post_percent

    # 사용자 정의 함수: Protocol 관련 Feature
    def extract_protocol_features(group):
        use_1_0 = any(group['Protocol'] == 'HTTP/1.0')
        return use_1_0

    # 사용자 정의 함수: Status 관련 Feature
    def extract_status_features(group):
        status_major_percent = len(group[group['Status'].isin(['200', '301', '302'])]) / max(1, len(group))
        status_404_percent = len(group[group['Status'] == '404']) / max(1, len(group))
        has_499 = any(group['Status'] == '499')
        status_cnt = group['Status'].nunique()
        return status_major_percent, status_404_percent, has_499, status_cnt

    # 사용자 정의 함수: Path 관련 Feature
    def extract_path_features(group):
        top1_path_cnt = group['Path'].value_counts().iloc[0] if not group['Path'].value_counts().empty else 0
        path_same = top1_path_cnt / max(1, len(group))
        path_xmlrpc = len(group[group['Path'].str.contains('xmlrpc.php') == True]) / max(1, len(group))
        return path_same, path_xmlrpc

    # 사용자 정의 함수: User Agent 관련 Feature
    def extract_ua_features(group):
        ua_cnt = group['UA'].nunique()
        return ua_cnt

    # 사용자 정의 함수: Payload 관련 Feature
    def extract_payload_features(group):
        has_payload = any(group['Payload'] != '-')
        return has_payload

    # 사용자 정의 함수: Bytes 관련 Feature
    def extract_bytes_features(group):
        bytes_avg = np.mean(group['Bytes'])
        bytes_std = np.std(group['Bytes'])
        return bytes_avg, bytes_std

    # Feature Engineering
    for entity in df['Host'].unique():
        group = df[df['Host'] == entity]

        # Method Features
        method_cnt, method_post_percent = extract_method_features(group)
        df.loc[df['Host'] == entity, 'method_cnt'] = method_cnt
        df.loc[df['Host'] == entity, 'method_post'] = method_post_percent

        # Protocol Features
        use_1_0 = extract_protocol_features(group)
        df.loc[df['Host'] == entity, 'protocol_1_0'] = use_1_0

        # Status Features
        status_major_percent, status_404_percent, has_499, status_cnt = extract_status_features(group)
        df.loc[df['Host'] == entity, 'status_major'] = status_major_percent
        df.loc[df['Host'] == entity, 'status_404'] = status_404_percent
        df.loc[df['Host'] == entity, 'status_499'] = has_499
        df.loc[df['Host'] == entity, 'status_cnt'] = status_cnt

        # Path Features
        path_same, path_xmlrpc = extract_path_features(group)
        df.loc[df['Host'] == entity, 'path_same'] = path_same
        df.loc[df['Host'] == entity, 'path_xmlrpc'] = path_xmlrpc

        # User Agent Features
        ua_cnt = extract_ua_features(group)
        df.loc[df['Host'] == entity, 'ua_cnt'] = ua_cnt

        # Payload Features
        has_payload = extract_payload_features(group)
        df.loc[df['Host'] == entity, 'has_payload'] = has_payload

        # Bytes Features
        bytes_avg, bytes_std = extract_bytes_features(group)
        df.loc[df['Host'] == entity, 'bytes_avg'] = bytes_avg
        df.loc[df['Host'] == entity, 'bytes_std'] = bytes_std

 # Bytes Features
    bytes_avg, bytes_std = extract_bytes_features(group)
    df.loc[df['Host'] == entity, 'bytes_avg'] = bytes_avg
    df.loc[df['Host'] == entity, 'bytes_std'] = bytes_std

    # Drop unwanted columns
    columns_to_drop = ['Unnamed: 0', 'Timestamp', 'Method', 'Protocol', 'Status', 'Referer', 'Path', 'Host', 'UA', 'Payload', 'Bytes']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    return df

# 사이드바에 링크 추가
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Contact"])

# 각 페이지에 대한 내용 표시
if page == "Home":
    st.title('로그 데이터 처리 앱')

    # 파일 업로드
    uploaded_file = st.file_uploader("CSV 파일 선택", type="csv")

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
    st.title('로그 데이터 처리 앱')

    # 파일 업로드
    uploaded_csvfile = st.file_uploader("CSV 파일 선택", type="csv")

    if uploaded_csvfile is not None:
        # CSV 파일 읽기
        df_entity = pd.read_csv(uploaded_csvfile)

        # Feature Extraction
        df_entity_processed = feature_extract(df_entity)

        # 전처리된 데이터 출력
        st.write("전처리된 데이터:")
        st.write(df_entity_processed)
elif page == "Contact":
    st.title('로그 데이터 처리 앱')

    # 파일 업로드
    uploaded_csvfile = st.file_uploader("CSV 파일 선택", type="csv")

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
