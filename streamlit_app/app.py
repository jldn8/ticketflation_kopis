import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings(action='ignore')
import os
import plotly.express as px
from pathlib import Path

# utils.py 불러오기
from utils import (
    ensure_columns,
    make_price_column,
    map_genre_4,
    ticketflation_by_year_max,
    priceband_rate_and_sales,
    add_musical_category
)

# -----------------------
# 기본 설정
# -----------------------
st.set_page_config(page_title="공연 데이터 대시보드", layout="wide")
st.title("공연 데이터 분석 및 정책 제안 대시보드")


# -----------------------
# 폰트 설정
# -----------------------
def set_font():
    BASE_DIR = Path(__file__).resolve().parent # app.py가 있는 폴더
    font_path = BASE_DIR / "fonts" / "NotoSansKR-Regular.ttf"

    fontprop = fm.FontProperties(fname=str(font_path))

    plt.rcParams["font.family"] = fontprop.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    
# -----------------------
# 데이터 로딩
# -----------------------
@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent # app.py가 있는 폴더
    DATA_DIR = BASE_DIR / "datasets" / "KOPIS"
    
    df = pd.read_csv(DATA_DIR / "performance_eda.csv")
    facility_df = pd.read_csv(DATA_DIR / "facility_df.csv")
    price_stats_df = pd.read_csv(DATA_DIR / "price_stats.csv")
    return df, facility_df, price_stats_df

df, facility_df, price_stats_df = load_data()
df = add_musical_category(df, facility_df)

# -----------------------
# 탭 구성
# -----------------------
tab1, tab2, tab3 = st.tabs(["분석 배경", "정책 제안", "정책 시뮬레이션"])

# -----------------------
# 탭 1. 분석 배경
# -----------------------

with tab1:
    st.header("분석 배경 설명")
    
    # -----------------------
    # 1) 장르별 평균 티켓 가격 비교
    # -----------------------
    st.subheader("장르별 평균 티켓 가격 비교")
    
    perf_for_price = make_price_column(df.copy()) # 대표 가격 칼럼 만들기
    perf_for_price = map_genre_4(perf_for_price)
    
    if ensure_columns(perf_for_price, ["대표가격", "장르4"], "장르별 평균 티켓 가격 비교"):
        min_year = int(perf_for_price["공연시작연도"].min())
        max_year = int(perf_for_price["공연시작연도"].max())
        
        year_range = st.slider(
            "연도 범위 선택",
            min_value = min_year,
            max_value = max_year,
            value=(min_year, max_year),
            step = 1
        )
        
        filtered = perf_for_price[
            (perf_for_price['공연시작연도'] >= year_range[0]) &
            (perf_for_price['공연시작연도'] <= year_range[1]) 
        ]
        
        avg_price = (
            filtered.dropna(subset=["대표가격", "장르4"])
                    .groupby("장르4")["대표가격"].mean()
        )
        
        # 1000원 단위 반올림
        avg_price = avg_price.round(-3).dropna()
        
        if not avg_price.empty:
            avg_price = avg_price.sort_values(ascending = True)
            # 가격 크기에 따라 색 농도 결정
            max_val = avg_price.max()
            min_val = avg_price.min()
            
            def get_color(val):
                # 0 ~ 1 정규화 후 20~80% 명도 사이로 매핑
                norm = (val - min_val) / (max_val - min_val + 1e-9)
                intensity = int(80 - norm * 60) # 값 클수록 어두워짐
                return f"rgba(33, 150, 243, {1 - norm*0.9})"  # 파란색 계열, 투명도로 진하기 표현
            
            cols = st.columns(len(avg_price))
            for col, (genre, price) in zip(cols, avg_price.items()):
                color = get_color(price)
                col.markdown(
                    f"""
                    <div style="
                        background-color:{color};
                        border-radius:12px;
                        padding:20px;
                        text-align:center;
                        color:white;
                        font-weight:bold;">
                        <div style="font-size:18px;">{genre}</div>
                        <div style="font-size:24px;">{int(price):,} 원</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("선택한 연도 범위에 해당하는 데이터가 없습니다.")

    # -----------------------
    # 2) 연도별 티켓플레이션 추이
    # -----------------------
    st.subheader("연도별 티켓플레이션 추이")
    
    perf_year_max = ticketflation_by_year_max(perf_for_price)
    
    if not perf_year_max.empty:
        df_long = perf_year_max.reset_index().melt(
            id_vars="공연시작연도",
            var_name="장르4",
            value_name="평균최대가격"
        )

        fig = px.line(
            df_long,
            x="공연시작연도",
            y="평균최대가격",
            color="장르4",
            markers=True
        )
        fig.update_layout(
            xaxis=dict(
                tickmode="linear",
                dtick=1,
                title="연도"
            ),
            yaxis=dict(
                title="평균 최대 티켓가격",
                tickformat=",.0f"  # 천 단위 콤마
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="center",
                x=0.5,
                title=None
            ),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("연도별 티켓플레이션 데이터를 표시할 수 없습니다.")
        
