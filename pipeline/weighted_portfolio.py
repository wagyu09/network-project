import pandas as pd
import numpy as np

# 가중치 조절 포트폴리오 설정 (노트북에서 추출)
SECTOR_COUNTS = {
    "Information Technology": 6,
    "Health Care": 5,
    "Consumer Staples": 4,
    "Utilities": 2,
}

def select_weighted_portfolio(test_prices: pd.DataFrame, sector_map: dict, random_state: int = 42) -> list:
    """
    주어진 섹터 비중(SECTOR_COUNTS)에 맞춰 포트폴리오 종목을 선택합니다.
    
    Args:
        test_prices (pd.DataFrame): 해당 분기의 주가 데이터 (티커가 컬럼)
        sector_map (dict): 티커를 키로, 섹터명을 값으로 하는 딕셔너리
        random_state (int): 재현성을 위한 시드 값

    Returns:
        list: 선택된 티커들의 리스트. 조건을 만족하지 못하면 빈 리스트 반환.
    """
    rng = np.random.default_rng(random_state)
    chosen_tickers = []
    
    # 데이터에 존재하는 유효한 티커들만 필터링
    valid_tickers_in_data = [t for t in test_prices.columns if not test_prices[t].isna().all()]
    
    for sector, count in SECTOR_COUNTS.items():
        # 해당 섹터이면서 데이터가 존재하는 티커 찾기
        sector_tickers = [
            t for t in valid_tickers_in_data 
            if sector_map.get(t) == sector
        ]
        
        # 종목 수가 부족하면 실패 처리
        if len(sector_tickers) < count:
            print(f"  Warning: Not enough tickers for sector '{sector}' (found {len(sector_tickers)}, need {count})")
            return []
            
        # 무작위 비복원 추출
        selected = rng.choice(sector_tickers, size=count, replace=False).tolist()
        chosen_tickers.extend(selected)
    
    return sorted(chosen_tickers)
