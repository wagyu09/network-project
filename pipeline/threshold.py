"""네트워크 엣지(간선)를 필터링하는 함수를 관리하는 모듈"""
from scipy import stats
import numpy as np
import pandas as pd

def threshold(corr_stats: pd.DataFrame, alpha: float = 0.05, c_min: float = 0.0) -> pd.DataFrame:
    """상관관계 통계량으로부터 통계적으로 유의미한 엣지만을 필터링

    피어슨 상관계수 'r'을 t-통계량으로 변환한 후, p-value를 계산하여
    주어진 유의수준(alpha)보다 낮은 p-value를 갖는 엣지만을 반환.
    추가적으로, 상관계수의 절댓값이 c_min 이상인 엣지만을 고려할 수 있음

    Args:
        corr_stats (pd.DataFrame): 'Correlation'과 'n'(관측치 수) 컬럼을
            포함하는 상관관계 통계 데이터프레임
        alpha (float): 필터링의 기준이 되는 유의수준. 이 값보다 작은
            p-value를 가진 엣지만 살아남음
        c_min (float): 살아남기 위해 필요한 최소 상관계수 절댓값

    Returns:
        pd.DataFrame: 통계적으로 유의미하다고 판단된 엣지(연결)들의
            정보가 담긴 데이터프레임
    """
    df = corr_stats.copy()
    r = df['Correlation'].to_numpy(float)
    n = df['n'].to_numpy(int)

    # 자유도(degree of freedom) 계산
    # 일반적인 피어슨 상관계수 t-검정의 자유도는 (n-2)
    # 하지만 이 프로젝트에서는 CAPM 모델의 잔차(residual)에 대한 상관관계를
    # 사용하므로, 모델의 파라미터 수(beta 1개)만큼 자유도를 추가로 차감하여 (n-3)을 사용
    dfree = n - 3
    # r의 값이 1 또는 -1일 경우 t-통계량 계산 시 0으로 나누는 오류가 발생하므로, 안전하게 값을 제한
    r_safe = np.clip(r, -0.999999, 0.999999)

    # t-통계량 계산. 상관계수 r을 t-분포상의 값으로 변환
    with np.errstate(divide='ignore', invalid='ignore'):
        tval = r_safe * np.sqrt(np.where(dfree > 0, dfree, np.nan) / (1.0 - r_safe**2))

    # p-value 계산. t-통계량과 자유도를 사용하여 양측 검정(two-sided test) p-value를 구함
    pval = 2.0 * stats.t.sf(np.abs(tval), df=np.where(dfree > 0, dfree, 1))
    # 자유도가 1보다 작거나 계산된 p-value가 유효하지 않은 경우, 유의하지 않음(p-value=1.0)으로 처리
    pval[(dfree < 1) | ~np.isfinite(pval)] = 1.0

    df['pval'] = pval

    # p-value가 유의수준(alpha)보다 작고, 상관계수 절대값이 c_min 이상인 엣지만 필터링
    p_edges = df[(df['pval'] < alpha) & (df['Correlation'].abs() >= c_min)].sort_index()

    return p_edges
