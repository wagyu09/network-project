def get_quarter_dates(quarter):
    """Converts a pandas Quarter object to start and end date strings."""
    start_date = quarter.start_time.strftime('%Y-%m-%d')
    end_date = quarter.end_time.strftime('%Y-%m-%d')
    return start_date, end_date
