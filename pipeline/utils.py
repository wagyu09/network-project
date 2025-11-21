"""프로젝트 전반에서 사용되는 범용 유틸리티 함수들을 관리하는 모듈"""

def get_quarter_dates(quarter):
    """Pandas의 Quarter 객체를 시작일과 종료일 문자열로 변환

    Args:
        quarter (pd.Period): Pandas의 분기 객체 (예: Period('2020Q1', 'Q-DEC'))

    Returns:
        tuple[str, str]: 해당 분기의 시작일과 종료일을 'YYYY-MM-DD' 형식의
            문자열로 담은 튜플
    """
    start_date = quarter.start_time.strftime('%Y-%m-%d')
    end_date = quarter.end_time.strftime('%Y-%m-%d')
    return start_date, end_date
