from PySide6.QtCore import QDate

def get_default_maturity_date() -> QDate:
    """
    Returns the default maturity date: today + 60 days.
    If the resulting date falls on a weekend, it is moved to the next Monday.
    """
    target_date = QDate.currentDate().addDays(60)
    day_of_week = target_date.dayOfWeek() # 1 = Monday, 6 = Saturday, 7 = Sunday
    if day_of_week == 6:
        target_date = target_date.addDays(2)
    elif day_of_week == 7:
        target_date = target_date.addDays(1)
    return target_date
