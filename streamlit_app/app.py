import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings(action='ignore')
import os

# -----------------------
# 기본 설정
# -----------------------
st.set_page_config(page_title="공연 데이터 대시보드", layout="wide")
st.title("공연 데이터 분석 및 정책 제안 대시보드")


# -----------------------
# 폰트 설정
# -----------------------
def set_font():
    font_dir = "./fonts"
    font_path = os.path.join(font_dir, "NotoSansKR-Regular.ttf")

    fontprop = fm.FontProperties(fname=font_path)

    plt.rcParams["font.family"] = fontprop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    
# -----------------------
# 데이터 로딩
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("./datasets/KOPIS/performance_eda.csv")
    facility_df = pd.read_csv("./datasets/KOPIS/facility_df.csv")
    price_stats_df = pd.read_csv("./datasets/KOPIS/price_stats.csv")
    return df, facility_df, price_stats_df

df, facility_df, price_stats_df = load_data()

# -----------------------
# 탭 구성
# -----------------------
tab1, tab2, tab3 = st.tabs(["분석 배경", "정책 제안", "정책 시뮬레이션"])

