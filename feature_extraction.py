import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import joblib

def feature_extract(df):
    # 사용할 특성 목록
    cols_to_train = ['Entity', 'Method', 'Protocol', 'Status', 'Referer', 'Path', 'UA', 'Payload', 'Bytes']

    # 사용자 정의 함수: Method 관련 특성
    def extract_method_features(group):
        method_cnt = group['Method'].nunique()
        method_post_percent = len(group[group['Method'] == 'POST']) / max(1, len(group))
        return method_cnt, method_post_percent

    # 사용자 정의 함수: Protocol 관련 특성
    def extract_protocol_features(group):
        use_1_0 = any(group['Protocol'] == 'HTTP/1.0')
        return use_1_0

    # 사용자 정의 함수: Status 관련 특성
    def extract_status_features(group):
        status_major_percent = len(group[group['Status'].isin(['200', '301', '302'])]) / max(1, len(group))
        status_404_percent = len(group[group['Status'] == '404']) / max(1, len(group))
        has_499 = any(group['Status'] == '499')
        status_cnt = group['Status'].nunique()
        return status_major_percent, status_404_percent, has_499, status_cnt

    # 사용자 정의 함수: Path 관련 특성
    def extract_path_features(group):
        top1_path_cnt = group['Path'].value_counts().iloc[0] if not group['Path'].value_counts().empty else 0
        path_same = top1_path_cnt / max(1, len(group))
        path_xmlrpc = len(group[group['Path'].str.contains('xmlrpc.php') == True]) / max(1, len(group))
        return path_same, path_xmlrpc

    # 사용자 정의 함수: User Agent 관련 특성
    def extract_ua_features(group):
        ua_cnt = group['UA'].nunique()
        return ua_cnt

    # 사용자 정의 함수: Payload 관련 특성
    def extract_payload_features(group):
        has_payload = any(group['Payload'] != '-')
        return has_payload

    # 사용자 정의 함수: Bytes 관련 특성
    def extract_bytes_features(group):
        bytes_avg = np.mean(group['Bytes'])
        bytes_std = np.std(group['Bytes'])
        return bytes_avg, bytes_std

    # 특성 공학
    for entity in df['Entity'].unique():
        group = df[df['Entity'] == entity]

        # Method 특성
        method_cnt, method_post_percent = extract_method_features(group)
        df.loc[df['Entity'] == entity, 'method_cnt'] = method_cnt
        df.loc[df['Entity'] == entity, 'method_post'] = method_post_percent

        # Protocol 특성
        use_1_0 = extract_protocol_features(group)
        df.loc[df['Entity'] == entity, 'protocol_1_0'] = use_1_0

        # Status 특성
        status_major_percent, status_404_percent, has_499, status_cnt = extract_status_features(group)
        df.loc[df['Entity'] == entity, 'status_major'] = status_major_percent
        df.loc[df['Entity'] == entity, 'status_404'] = status_404_percent
        df.loc[df['Entity'] == entity, 'status_499'] = has_499
        df.loc[df['Entity'] == entity, 'status_cnt'] = status_cnt

        # Path 특성
        path_same, path_xmlrpc = extract_path_features(group)
        df.loc[df['Entity'] == entity, 'path_same'] = path_same
        df.loc[df['Entity'] == entity, 'path_xmlrpc'] = path_xmlrpc

        # User Agent 특성
        ua_cnt = extract_ua_features(group)
        df.loc[df['Entity'] == entity, 'ua_cnt'] = ua_cnt

        # Payload 특성
        has_payload = extract_payload_features(group)
        df.loc[df['Entity'] == entity, 'has_payload'] = has_payload

        # Bytes 특성
        bytes_avg, bytes_std = extract_bytes_features(group)
        df.loc[df['Entity'] == entity, 'bytes_avg'] = bytes_avg
        df.loc[df['Entity'] == entity, 'bytes_std'] = bytes_std

    # 불필요한 열 제거
    columns_to_drop = ['Unnamed: 0', 'Timestamp', 'Method', 'Protocol', 'Status', 'Referer', 'Path', 'UA', 'Payload', 'Bytes']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    return df


def main():
    st.title('로그 데이터 처리 앱')

    # 파일 업로드
    uploaded_csvfile = st.file_uploader("CSV 파일 선택", type="csv")

    if uploaded_csvfile is not None:
        # CSV 파일 읽기
        df_entity = pd.read_csv(uploaded_csvfile)

        # 특성 추출
        df_entity_processed = feature_extract(df_entity)

        # 전처리된 데이터 출력
        st.subheader("전처리된 데이터")
        st.dataframe(df_entity_processed)

if __name__ == '__main__':
    main()
