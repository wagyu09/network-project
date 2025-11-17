from data_loader import DataLoader
from corr_calculator import corr_calculator
from threshold import threshold

STARTDATE = '2024-01-01'
ENDDATE = '2025-01-01'

data_loader = DataLoader(STARTDATE, ENDDATE)
daily, monthly, mkt_idx = data_loader.load_data()

corr_data, resid_df = corr_calculator(daily, mkt_idx)
p_edges = threshold(corr_data, resid_df)

