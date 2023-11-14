import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def main():
    # st.title('로그 데이터 처리 앱')

    # 파일 업로드
    # uploaded_csvfile = st.file_uploader("CSV 파일 선택", type="csv")

    # if uploaded_csvfile is not None:
    # CSV 파일 읽기
    df_entity = pd.read_csv('train_processed.csv')
    

    # 선택할 feature들
    cols_to_train = ['method_cnt', 'method_post', 'protocol_1_0', 'status_major', 'status_404', 'status_499', 'status_cnt',
                     'path_same', 'path_xmlrpc', 'ua_cnt', 'has_payload', 'bytes_avg', 'bytes_std']
    
    # PCA를 사용하여 데이터의 차원을 2로 축소
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_entity[cols_to_train])
    
    # PCA 결과를 데이터프레임에 추가
    df_entity['pca_1'] = pca_result[:, 0]
    df_entity['pca_2'] = pca_result[:, 1]
    
    # Streamlit 앱 레이아웃
    st.title("전체 Feature를 이용한 이상탐지된 Entity 시각화 (PCA 결과)")
    
    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df_entity['pca_1'], df_entity['pca_2'], c=df_entity['cluster_kmeans'], cmap='viridis', s=60)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("전체 Feature를 이용한 이상탐지된 Entity 시각화 (PCA 결과)")
    ax.legend(*scatter.legend_elements(), title='클러스터')
    
    # Streamlit에 Matplotlib 그래프 표시
    st.pyplot(fig)
   
        

if __name__ == '__main__':
    main()
