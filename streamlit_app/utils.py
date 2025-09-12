import streamlit as st
import pandas as pd
import numpy as np

# -----------------------
# 전처리/유틸 함수
# -----------------------

def ensure_columns(df: pd.DataFrame, cols: list[str], where: str=""):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.warning(f"[{where}] 누락 컬럼: {missing} → 관련 차트는 생략됩니다.")
        return False
    return True


def make_price_column(df: pd.DataFrame) -> pd.DataFrame:
    """대표 가격 컬럼 생성: 최대 가격 > 최소 가격 > 티켓가격 순으로 사용"""
    fallback_price = df.get("티켓가격", pd.Series([np.nan]*len(df)))
    df["대표가격"] = np.where(df.get("최대가격").notna(), df["최대가격"],
                          np.where(df.get("최소가격").notna(), df["최소가격"],
                                   fallback_price))
    df["대표가격"] = pd.to_numeric(df["대표가격"], errors="coerce")
    return df

def categorize_musical(row):
    """
    장르가 '뮤지컬'일 때만 세부 구분
    - 1000석 이상: 뮤지컬(대형)
    - 1000석 미만: 뮤지컬(중소)
    """
    
    if "뮤지컬" not in str(row.get("공연장르명", "")):
        return None
    
    seats = row.get("객석수", None)
    
    try:
        seats = int(seats)
    except (ValueError, TypeError):
        return "뮤지컬(기타)"
    
    return "뮤지컬(대형)" if seats >= 1000 else "뮤지컬(중소)"

def add_musical_category(df_perf: pd.DataFrame, df_fac: pd.DataFrame) -> pd.DataFrame:
    """
    공연 데이터(performance_eda) + 시설 데이터(facility_df)를 조인 후
    뮤지컬 규모 구분 컬럼('뮤지컬_구분') 추가
    """
    merged = df_perf.merge(
        df_fac[["공연시설ID", "객석수"]],
        on="공연시설ID",
        how="left"
    )
    merged["뮤지컬_구분"] = merged.apply(categorize_musical, axis=1)
    return merged

def normalize_genre(row):
    g = str(row["공연장르명"])

    if g == "연극":
        return "연극"
    elif g == "대중음악":
        return "케이팝"
    elif g == "뮤지컬":
        if row["뮤지컬_구분"] in ["뮤지컬(대형)", "뮤지컬(중소)"]:
            return row["뮤지컬_구분"]
        else:
            return np.nan  # 기타 뮤지컬은 제외
    else:
        return np.nan
        
def map_genre_4(df: pd.DataFrame) -> pd.DataFrame:
    """
    최종 4개 카테고리로 매핑:
    - 연극 → '연극'
    - 대중음악 → '케이팝'
    - 뮤지컬 → 뮤지컬_구분 기반 (대형, 창작, 소규모)
      단, '뮤지컬(기타)'는 제외
    """
    
    df['뮤지컬_구분'] = df.apply(categorize_musical, axis=1)        
    df["장르4"] = df.apply(normalize_genre, axis=1)
    
    return df
        
def ticketflation_by_year(df_perf: pd.DataFrame) -> pd.DataFrame:
    """
    연도 x 장르4 평균 가격(대표가격) 피벗
    """
    if not ensure_columns(df_perf, ["공연시작연도", "대표가격", "장르4"], "연도별 티켓플레이션(대표가격)"):
        return pd.DataFrame()
    
    d = df_perf.dropna(subset=["공연시작연도", "대표가격", "장르4"])
    gp = d.groupby(["공연시작연도", "장르4"])["대표가격"].mean().reset_index()
    pivot = gp.pivot(index="공연시작연도", columns="장르4", values="대표가격").sort_index()
    return pivot

def ticketflation_by_year_max(df_perf: pd.DataFrame) -> pd.DataFrame:
    """
    연도 x 장르4 평균 가격(최대가격) 피벗
    티켓플레이션 추이를 고가 정책 기준으로 보기 위함
    """
    if not ensure_columns(df_perf, ["공연시작연도", "대표가격", "장르4"], "연도별 티켓플레이션(최대가격)"):
        return pd.DataFrame()
    
    d = df_perf.dropna(subset=["공연시작연도", "최대가격", "장르4"])
    gp = d.groupby(["공연시작연도", "장르4"])["최대가격"].mean().reset_index()
    pivot = gp.pivot(index="공연시작연도", columns="장르4", values="최대가격").sort_index()
    return pivot

def sort_priceband(df: pd.DataFrame) -> pd.DataFrame:
    """
    가격대 순서 지정 정렬 
    """
    if df.empty: return df
    
    order = [
        "0원",
        "3만원미만",
        "3만원이상~5만원미만",
        "5만원이상~7만원미만",
        "7만원이상~10만원미만",
        "10만원이상~15만원미만",
        "15만원이상"
    ]
    
    idx = [x for x in order if x in df.index]
    idx += [x for x in df.index if x not in idx]
    return df.loc[idx]
    

def priceband_rate_and_sales(df_price: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    price_stats.csv 기반:
    - 예매율 = 예매수 / 총티켓판매수
    - 가격대 x 장르 피벗 (예매율 / 총티켓판매수)
    """
    
    if not ensure_columns(df_price, ["장르", "예매수", "총티켓판매수", "가격대"], "가격대 지표"):
        return pd.DataFrame(), pd.DataFrame()
    
    d = df_price.copy()
    # 0 division 방지
    d["예매수"] = pd.to_numeric(d["예매수"], errors="coerce")
    d["총티켓판매수"] = pd.to_numeric(d["총티켓판매수"], errors="coerce")
    d = d[d["총티켓판매수"] > 0]
    
    d["예매율"] = d["예매수"] / d["총티켓판매수"] * 100.0
    
    d["장르4"] = d["장르"].apply(normalize_genre)
    d = d.dropna(subset=["장르4", "가격대"])
    
    rate_pivot = d.pivot_table(index="가격대", columns="장르4", values="예매율", aggfunc="mean")
    sales_pivot = d.pivot_table(index="가격대", columns="장르4", values="총티켓판매수", aggfunc="sum")
    
    return sort_priceband(rate_pivot), sort_priceband(sales_pivot)
    