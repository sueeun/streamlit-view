import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib

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


def main():
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

if __name__ == '__main__':
    main()
