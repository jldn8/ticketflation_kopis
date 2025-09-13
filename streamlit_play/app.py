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
import altair as alt


# ======= 데이터 로딩 ========
@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent # app.py가 있는 폴더
    DATA_DIR = BASE_DIR / "datasets"
    
    price_stats = pd.read_csv(DATA_DIR / "가격대별예매통계.csv")
    genre_stats = pd.read_csv(DATA_DIR / "장르별예매통계.csv")
    genre_service = pd.read_csv(DATA_DIR / "장르별통계목록.csv")
    time_stats = pd.read_csv(DATA_DIR / "시간대별예매통계.csv")
    perf_detail = pd.read_csv(DATA_DIR / "연극상세정보(시간대+가격대).csv")
    return price_stats, genre_stats, genre_service, time_stats, perf_detail

price_stats, genre_stats, genre_service, time_stats, perf_detail = load_data()

@st.cache_data
def load_id_date_data():
    BASE_DIR = Path(__file__).resolve().parent # app.py가 있는 폴더
    DATA_DIR = BASE_DIR / "datasets"
    
    return pd.read_csv(DATA_DIR / "상연일자별공연ID목록.csv")

@st.cache_data
def load_facility_2324_data():
    BASE_DIR = Path(__file__).resolve().parent # app.py가 있는 폴더
    DATA_DIR = BASE_DIR / "datasets"
    
    return pd.read_csv(DATA_DIR / "공연시설상세정보(2324연극).csv")
    

# 날짜 처리
price_stats["날짜"] = pd.to_datetime(price_stats['날짜'], format="%Y%m%d")
genre_stats["날짜"] = pd.to_datetime(genre_stats['날짜'], format="%Y%m%d")
time_stats["날짜"] = pd.to_datetime(time_stats['날짜'], format="%Y%m%d")

price_stats["총 티켓판매액"] = price_stats["총 티켓판매액"] * 1000

day_map = {
    "Monday": "월요일",
    "Tuesday": "화요일",
    "Wednesday": "수요일",
    "Thursday": "목요일",
    "Friday": "금요일",
    "Saturday": "토요일",
    "Sunday": "일요일",
}
    

st.title("연극 분석 대시보드")

tab1, tab2 = st.tabs(["시기별", "시간대별"])

with tab1:
    st.header("시기별 추이 분석")
    
    # --- 선택 옵션 ---
    chart_type = st.radio(
        "분석 단위 선택",
        ["월별", "요일별", "분기별", "계절별"],
        horizontal=True,
        key="period_chart_type"
    )
    
    sort_option = st.radio(
        "정렬 방식 선택",
        ["기본순", "낮은순", "높은순"],
        horizontal=True,
        key="period_sort_option"
    )
        
    # --- 데이터 가공 ---
    df = price_stats.copy()
    
    df = df[df["장르"] == "연극"].copy()
    
    # 가격대 필터 옵션 추가
    price_options = df["가격대"].unique().tolist()
    selected_prices = st.multiselect(
        "가격대 선택",
        options=price_options,
        default=["5~7", "7~10"]
    )
    st.caption("※ 아무것도 선택하지 않으면 전체 가격대가 자동 적용됩니다.")

    # 아무것도 선택되지 않은 경우 → 전체선택
    if not selected_prices:
        selected_prices = price_options
    
    # 가격대 필터링 적용
    df = df[df["가격대"].isin(selected_prices)].copy()
    
    df["연도"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month
    
    df["요일"] = df["날짜"].dt.day_name().map(day_map)

    df["분기"] = df["날짜"].dt.to_period("Q").astype(str)

    month = df["날짜"].dt.month
    conditions = [
        month.isin([3,4,5]),
        month.isin([6,7,8]),
        month.isin([9,10,11]),
        month.isin([12,1,2])
    ]
    choices = ["봄","여름","가을","겨울"]

    df["계절"] = np.select(conditions, choices)
    
    # --- 단위별 집계 ---
    if chart_type == "월별":
        df_filtered = df[df["연도"].isin([2023, 2024])]
        grouped = (
            df_filtered.groupby(["연도", "월"])[["총 티켓판매수","총 티켓판매액"]]
            .sum()
            .groupby("월").mean()
            .reset_index()
        )
        grouped["월"] = grouped["월"].astype(str) + "월"
        x_col = "월"
    
    elif chart_type == "분기별":
        df_filtered = df[df["연도"].isin([2023, 2024])]
        
        df_filtered["분기"] = "Q" + df_filtered["날짜"].dt.quarter.astype(str)
        
        grouped = (
            df_filtered.groupby(["연도","분기"])[["총 티켓판매수","총 티켓판매액"]]
            .sum()
            .groupby("분기").mean()
            .reset_index()
        )
        
        x_col = "분기"
    
    elif chart_type == "요일별":
        grouped = df.groupby("요일")[["총 티켓판매수","총 티켓판매액"]].sum().reset_index()
        order = ["월요일","화요일","수요일","목요일","금요일","토요일","일요일"]
        grouped["요일"] = pd.Categorical(grouped["요일"], categories=order, ordered=True)
        grouped = grouped.sort_values("요일")
        x_col = "요일"
    
    else:  # 계절별
        grouped = df.groupby("계절")[["총 티켓판매수","총 티켓판매액"]].sum().reset_index()
        order = ["봄","여름","가을","겨울"]
        grouped["계절"] = pd.Categorical(grouped["계절"], categories=order, ordered=True)
        grouped = grouped.sort_values("계절")
        x_col = "계절"
    
    # --- 정렬 옵션 반영 ---
    if sort_option == "낮은순":
        grouped = grouped.sort_values("총 티켓판매수", ascending=True)
    elif sort_option == "높은순":
        grouped = grouped.sort_values("총 티켓판매수", ascending=False)
    # "기본순"은 그대로 둠
    
    # --- 차트 그리기 ---
    chart = alt.Chart(grouped).mark_bar().encode(
        x=alt.X(x_col, sort=list(grouped[x_col]), axis=alt.Axis(labelAngle=0)),
        y="총 티켓판매수:Q",
        color=alt.Color(
            "총 티켓판매수:Q",
            scale=alt.Scale(scheme="blues"),
            legend=None
        ),
        tooltip=[x_col, "총 티켓판매수", "총 티켓판매액"]
    ).properties(title=f"{chart_type} 티켓판매수")
    
    st.altair_chart(chart, use_container_width=True)


    ####### ------------------ #######
    
    df_daily = df.groupby("날짜")[["총 티켓판매수", "총 티켓판매액"]].sum().reset_index()
    df_daily["연도"] = df_daily["날짜"].dt.year
    df_daily["월"] = df_daily["날짜"].dt.month
    df_daily["분기"] = "Q" + df_daily["날짜"].dt.quarter.astype(str)
    month = df_daily["날짜"].dt.month
    conditions = [
        month.isin([3,4,5]),
        month.isin([6,7,8]),
        month.isin([9,10,11]),
        month.isin([12,1,2])
    ]
    choices = ["봄","여름","가을","겨울"]

    df_daily["계절"] = np.select(conditions, choices)
    
    df_daily["요일"] = df_daily["날짜"].dt.day_name(locale="ko_KR")
    
    card_type = st.radio("카드 단위 선택", ["월별","요일별","분기별","계절별"], horizontal=True)

    if card_type == "월별":
        df_group = df_daily.groupby("월")[["총 티켓판매수","총 티켓판매액"]].mean().reset_index()
        options = sorted(df_group["월"].unique())
        selected = st.multiselect("월 선택", options, default=options, format_func=lambda x: f"{x}월")

    elif card_type == "요일별":
        order = ["월요일","화요일","수요일","목요일","금요일","토요일","일요일"]
        df_group = df_daily.groupby("요일")[["총 티켓판매수","총 티켓판매액"]].mean().reset_index()
        df_group["요일"] = pd.Categorical(df_group["요일"], categories=order, ordered=True)
        df_group = df_group.sort_values("요일")
        selected = st.multiselect("요일 선택", order, default=order)

    elif card_type == "분기별":
        order = ["Q1","Q2","Q3","Q4"]
        df_group = df_daily.groupby("분기")[["총 티켓판매수","총 티켓판매액"]].mean().reset_index()
        df_group["분기"] = pd.Categorical(df_group["분기"], categories=order, ordered=True)
        df_group = df_group.sort_values("분기")
        selected = st.multiselect("분기 선택", order, default=order)

    else:  # 계절별
        order = ["봄","여름","가을","겨울"]
        df_group = df_daily.groupby("계절")[["총 티켓판매수","총 티켓판매액"]].mean().reset_index()
        df_group["계절"] = pd.Categorical(df_group["계절"], categories=order, ordered=True)
        df_group = df_group.sort_values("계절")
        selected = st.multiselect("계절 선택", order, default=order)

    # 선택 적용
    st.caption("※ 선택하지 않으면 전체 적용됩니다.")
    
    if not selected:
        selected = df_group.iloc[:,0].unique()  # 첫 번째 컬럼 기준 전체
    filtered = df_group[df_group.iloc[:,0].isin(selected)]

    if not filtered.empty:
        avg_sales = filtered["총 티켓판매수"].mean()
        avg_revenue = filtered["총 티켓판매액"].mean()
        avg_price = avg_revenue / avg_sales if avg_sales > 0 else 0
    else:
        avg_sales, avg_revenue, avg_price = 0, 0, 0

    col1, col2, col3 = st.columns(3)
    col1.metric("평균 티켓판매수 (하루)", f"{avg_sales:,.0f}")
    col2.metric("평균 티켓판매액 (하루)", f"{avg_revenue:,.0f} 원")
    col3.metric("평균 티켓가격", f"{avg_price:,.0f} 원")
        
    ####### ------------------ #######

    st.markdown("---")

    # 1. 최고 티켓판매 bar 찾기
    top_cat = grouped.loc[grouped["총 티켓판매수"].idxmax(), x_col]
    st.write(f"### {chart_type} 중 티켓 판매수 최고: {top_cat}")
    
    view_option = st.radio(
        "보기 옵션",
        ["공연", "공연 시설"],
        horizontal=True
    )
    
    # 2. 해당 카테고리 날짜 필터링
    if chart_type == "월별":
        top_dates = df[df["월"] == int(str(top_cat).replace("월",""))]["날짜"].unique()

    elif chart_type == "요일별":
        top_dates = df[df["요일"] == top_cat]["날짜"].unique()

    elif chart_type == "분기별":
        # top_cat이 "Q1" 형식이므로 숫자만 추출
        q_num = int(top_cat.replace("Q",""))
        top_dates = df[df["날짜"].dt.quarter == q_num]["날짜"].unique()

    else:  # 계절별
        top_dates = df[df["계절"] == top_cat]["날짜"].unique()

    # 날짜를 YYYYMMDD 정수로 변환
    top_dates_fmt = pd.to_datetime(top_dates).strftime("%Y%m%d").astype(int)

    # 3. 상연일자별 공연ID 매핑
    id_date_df = load_id_date_data()
    id_date_df["상연일"] = id_date_df["상연일"].astype(int)
    id_date_df["공연ID"] = id_date_df["공연ID"].astype(str)

    play_ids = id_date_df[id_date_df["상연일"].isin(top_dates_fmt)]["공연ID"].unique()

    # 4. 상세정보 조인 (중복 제거 포함)
    perf_detail["공연ID"] = perf_detail["공연ID"].astype(str)
    top_perfs = perf_detail[perf_detail["공연ID"].isin(play_ids)]

    # 가격대 필터 적용
    top_perfs = top_perfs[top_perfs["가격대"].isin(selected_prices)]

    # --- 요일별일 경우, 해당 요일을 요일시간에 반드시 포함하도록 필터 ---
    if chart_type == "요일별":
        target_day = top_cat.replace("요일", "")  # 예: "금요일" → "금"
        top_perfs = top_perfs[top_perfs["요일시간"].astype(str).str.contains(target_day)]

    # 공연ID 기준으로 중복 제거
    top_perfs_unique = top_perfs.drop_duplicates(subset=["공연ID"])
    
    # --- 5. 출력 (보기 옵션에 따라 다르게) ---
    if top_perfs_unique.empty:
        st.warning(f"{top_cat}에 해당하는 데이터가 없습니다.")
    else:
        st.info(f"**{top_cat}에 열린 연극**의 {view_option} 정보입니다. 기획에 참고하세요.")
        if view_option == "공연":
            st.dataframe(
                top_perfs_unique[
                    ["공연명","티켓가격_list","요일", "공연시각", "런타임_분", "공연시설명","출연진","제작사"]
                ]
            )
        else:  # 공연 시설 보기
            facility_df = load_facility_2324_data()  # 공연시설상세정보.csv 로딩 함수 필요
            
            merged_fac = facility_df.merge(
                top_perfs_unique[["공연시설ID"]],
                on="공연시설ID",
                how="inner"
            ).drop_duplicates(subset=["공연시설ID"])

            merged_fac["객석수"] = pd.to_numeric(merged_fac["객석수"], errors="coerce")

            st.dataframe(
                merged_fac[["공연시설명","시도","시군구","객석수"]]
                    .sort_values("객석수", ascending=False)
            )
            
            
with tab2:
    st.header("시간대별 분석")
    
    st.info("※ 시간대별 분석은 **전체 장르가 합쳐진 통계 자료**를 기반으로 합니다. 참고용으로 활용하세요.")
    
    # --- 선택 옵션 ---
    chart_type = st.radio(
        "분석 단위 선택",
        ["시간대_그룹", "시간대_대분류"],
        horizontal=True,
        key="time_chart_type"
    )
    
    sort_option = st.radio(
        "정렬 방식 선택",
        ["기본순", "낮은순", "높은순"],
        horizontal=True,
        key="time_sort_option"
    )
    
    # --- 데이터 준비 ---
    df = time_stats.copy()
    df = df[df['시간대_그룹'] != "nan"].copy()
    df["연도"] = df["날짜"].dt.year
    df["월"] = df["날짜"].dt.month
        
    # --- 집계 ---
    if chart_type == "시간대_그룹":
        grouped = df.groupby("시간대_그룹")[["총 티켓판매수","총 티켓판매액"]].sum().reset_index()
        x_col = "시간대_그룹"

        grouped["start_hour"] = grouped[x_col].str.extract(r"^(\d+)").astype(int)
        grouped = grouped.sort_values("start_hour").drop(columns="start_hour")

    else:  # 시간대_대분류
        grouped = df.groupby("시간대_대분류")[["총 티켓판매수","총 티켓판매액"]].sum().reset_index()
        x_col = "시간대_대분류"

        order = ["주간","저녁","심야"]
        grouped[x_col] = pd.Categorical(grouped[x_col], categories=order, ordered=True)
        grouped = grouped.sort_values(x_col)
    
    # --- 정렬 옵션 반영 ---
    if sort_option == "낮은순":
        grouped = grouped.sort_values("총 티켓판매수", ascending=True)
    elif sort_option == "높은순":
        grouped = grouped.sort_values("총 티켓판매수", ascending=False)
    
    # --- 바 차트 ---
    chart = alt.Chart(grouped).mark_bar().encode(
        x=alt.X(x_col, sort=list(grouped[x_col].astype(str))),
        y="총 티켓판매수:Q",
        color=alt.Color(
            "총 티켓판매수:Q",
            scale=alt.Scale(scheme="blues"),
            legend=None
        ),
        tooltip=[x_col, "총 티켓판매수", "총 티켓판매액"]
    ).properties(title=f"{chart_type}별 티켓판매수")
    
    st.altair_chart(chart, use_container_width=True)

    # --- 월별 히트맵 ---
    if chart_type == "시간대_그룹":
        heatmap_data = df.groupby(["월","시간대_그룹"])["총 티켓판매수"].sum().reset_index()
        heatmap_data["start_hour"] = heatmap_data["시간대_그룹"].str.extract(r"^(\d+)").astype(int)
        heatmap_data = heatmap_data[heatmap_data["start_hour"] >= 9]
        time_order = heatmap_data.sort_values("start_hour")["시간대_그룹"].unique().tolist()

        chart = alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X("월:O", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("시간대_그룹:O", sort=time_order),
            color=alt.Color("총 티켓판매수:Q",
                scale=alt.Scale(type="sqrt", scheme="blues"),
                legend=alt.Legend(format="~s", title="총 티켓판매수")),
            tooltip=["월","시간대_그룹","총 티켓판매수"]
        ).properties(title="월별 × 시간대_그룹 티켓판매 히트맵")

    else:  # 시간대_대분류
        heatmap_data = df.groupby(["월","시간대_대분류"])["총 티켓판매수"].sum().reset_index()
        order_time = ["주간","저녁","심야"]
        heatmap_data["시간대_대분류"] = pd.Categorical(heatmap_data["시간대_대분류"], categories=order_time, ordered=True)

        chart = alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X("월:O", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("시간대_대분류:O", sort=order_time),
            color=alt.Color("총 티켓판매수:Q",
                scale=alt.Scale(type="linear", scheme="blues"),
                legend=alt.Legend(format="~s", title="총 티켓판매수")),
            tooltip=["월","시간대_대분류","총 티켓판매수"]
        ).properties(title="월별 × 시간대_대분류 티켓판매 히트맵")

    st.altair_chart(chart, use_container_width=True)

    # --- 요일별 히트맵 ---
    df["요일"] = df["날짜"].dt.day_name().map(day_map)
    order_day = ["월요일","화요일","수요일","목요일","금요일","토요일","일요일"]
    df["요일"] = pd.Categorical(df["요일"], categories=order_day, ordered=True)

    if chart_type == "시간대_그룹":
        heatmap_data = df.groupby(["요일","시간대_그룹"])["총 티켓판매수"].sum().reset_index()
        heatmap_data["start_hour"] = heatmap_data["시간대_그룹"].str.extract(r"^(\d+)").astype(int)
        heatmap_data = heatmap_data[heatmap_data["start_hour"] >= 9]
        time_order = heatmap_data.sort_values("start_hour")["시간대_그룹"].unique().tolist()

        chart = alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X("요일:O", sort=order_day, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("시간대_그룹:O", sort=time_order),
            color=alt.Color("총 티켓판매수:Q",
                scale=alt.Scale(type="sqrt", scheme="blues"),
                legend=alt.Legend(format="~s", title="총 티켓판매수")),
            tooltip=["요일","시간대_그룹","총 티켓판매수"]
        ).properties(title="요일별 × 시간대_그룹 티켓판매 히트맵")

    else:  # 시간대_대분류
        heatmap_data = df.groupby(["요일","시간대_대분류"])["총 티켓판매수"].sum().reset_index()
        order_time = ["주간","저녁","심야"]
        heatmap_data["시간대_대분류"] = pd.Categorical(heatmap_data["시간대_대분류"], categories=order_time, ordered=True)

        chart = alt.Chart(heatmap_data).mark_rect().encode(
            x=alt.X("요일:O", sort=order_day, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("시간대_대분류:O", sort=order_time),
            color=alt.Color("총 티켓판매수:Q",
                scale=alt.Scale(type="linear", scheme="blues"),
                legend=alt.Legend(format="~s", title="총 티켓판매수")),
            tooltip=["요일","시간대_대분류","총 티켓판매수"]
        ).properties(title="요일별 × 시간대_대분류 티켓판매 히트맵")

    st.altair_chart(chart, use_container_width=True)    
    
    # ----------------------------------
    
    st.markdown("---")
    
    # --- 추가 카드: 최고 시간대 ---
    if not grouped.empty:
        top_cat = grouped.loc[grouped["총 티켓판매수"].idxmax(), x_col]
        st.write(f"### {chart_type} 중 티켓 판매수 최고: {top_cat}")

        view_option = st.radio(
            "보기 옵션",
            ["공연", "공연 시설"],
            horizontal=True,
            key="time_view_option"
        )
        
        def to_daypart(x):
            if pd.isna(x):
                return np.nan
            # "0-3", "3-6" 같은 문자열을 받아서 주간/저녁/심야로 변환
            start = int(x.split("-")[0])  # 시작 시간
            if 9 <= start < 18:
                return "주간"
            elif 18 <= start < 24:
                return "저녁"
            else:
                return "심야"
        
        perf_detail = perf_detail.copy()
        if "시간대_그룹" in perf_detail.columns:
            perf_detail["시간대_대분류"] = perf_detail["시간대_그룹"].apply(to_daypart)

        if chart_type == "시간대_그룹":
            top_perfs = perf_detail[perf_detail["시간대_그룹"] == top_cat]
        else:  # 시간대_대분류
            top_perfs = perf_detail[perf_detail["시간대_대분류"] == top_cat]

        # 가격대 필터 적용
        top_perfs = top_perfs[top_perfs["가격대"].isin(selected_prices)]

        # 공연ID 기준 중복 제거
        top_perfs_unique = top_perfs.drop_duplicates(subset=["공연ID"])

        if top_perfs_unique.empty:
            st.warning(f"{top_cat}에 해당하는 데이터가 없습니다.")
        else:
            st.info(f"**{top_cat}에 열린 연극**의 {view_option} 정보입니다. 기획에 참고하세요.")
            
            if view_option == "공연":
                st.dataframe(
                    top_perfs_unique[
                        ["공연명","티켓가격_list","요일", "공연시각", "런타임_분","공연시설명","출연진","제작사"]
                    ]
                )
            else:  # 공연 시설 보기
                facility_df = load_facility_2324_data()
                merged_fac = facility_df.merge(
                    top_perfs_unique[["공연시설ID"]],
                    on="공연시설ID",
                    how="inner"
                ).drop_duplicates(subset=["공연시설ID"])

                merged_fac["객석수"] = pd.to_numeric(merged_fac["객석수"], errors="coerce")

                st.dataframe(
                    merged_fac[["공연시설명","시도","시군구","객석수"]]
                        .sort_values("객석수", ascending=False)
                )