from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

import pandas as pd
import xarray as xr
import abc


class AbstractYear(abc.ABC):
    month_dict: Dict[str, int]

    @abc.abstractmethod
    def to_datetime(self):
        pass

    @staticmethod
    def year_month_to_datetime(year: int, month: str):
        return datetime.strptime(f"{str(year)} {month}", "%Y %B")

    def __len__(self) -> int:
        return len(self.month_dict)

    def select_months(self, months: List[str]):
        self.month_dict = {k: self.month_dict[k] for k in months}


class Year(AbstractYear):
    def __init__(self, year: int | None = None) -> None:
        self.month_dict = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        if year is not None:
            self.year = year
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        return "Year " + str(self.year)

    def as_dict(self):
        return {f"{str(self.year)}": self.month_dict}

    def to_datetime(self):
        return [self.year_month_to_datetime(self.year, month) for month in self.month_dict]

    def to_xarray_coord(self):
        return xr.Coordinates({"time": self.to_datetime()})


class WinterYear(AbstractYear):
    def __init__(self, from_year: int, to_year: int) -> None:
        self.month_dict = {
            "october": 1,
            "november": 2,
            "december": 3,
            "january": 4,
            "february": 5,
            "march": 6,
            "april": 7,
            "may": 8,
            "june": 9,
            "july": 10,
            "august": 11,
            "september": 12,
        }

        self.n_of_days_dict = {
            "october": 31,
            "november": 30,
            "december": 31,
            "january": 31,
            "february": 28,
            "march": 31,
            "april": 30,
            "may": 31,
            "june": 30,
            "july": 31,
            "august": 31,
            "september": 30,
        }
        self.from_year = from_year
        self.to_year = to_year
        if to_year % 4 == 0:
            self.n_of_days_dict.update(february=29)

    def __repr__(self) -> str:
        return "Winter Year " + str(self.from_year) + "/" + str(self.to_year)

    @property
    def days_per_month(self):
        return np.array(list(self.n_of_days_dict.values()))

    def iterate_days(self):
        for day in pd.date_range(start=f"{self.from_year}/10/01", end=f"{self.to_year}/09/30", freq="D"):
            yield day

    def to_datetime(self):
        out = [
            Year.year_month_to_datetime(self.from_year, month)
            for month in self.month_dict
            if month in ("october", "november", "december")
        ]
        out.extend(
            [
                Year.year_month_to_datetime(self.to_year, month)
                for month in self.month_dict
                if month in ("january", "february", "march", "april", "may", "june", "july", "august", "september")
            ]
        )
        return out

    def to_tuple(self) -> Tuple[int, int]:
        return (self.from_year, self.to_year)
