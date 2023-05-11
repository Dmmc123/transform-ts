from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.varmax import VARMAX
from sktime.forecasting.ets import AutoETS

import pickle
from pathlib import Path

from abc import ABC, abstractmethod

import pandas as pd


class ZooModel(ABC):

    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, days: int) -> pd.Series:
        pass

    @abstractmethod
    def save(self, weights_dir: str, company_name: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def from_weights(cls, weights_dir: str, company_name: str) -> "ZooModel":
        pass


class ZooVARMAX(ZooModel):
    def __init__(self, predictor = None):
        self.predictor = predictor

    def train(self, data: pd.DataFrame) -> None:
        self.predictor = VARMAX(maxiter=20)
        self.predictor.fit(y=data)

    def predict(self, days: int) -> pd.Series:
        return self.predictor.predict(fh=list(range(1, days+1)))["Close"]

    def save(self, weights_dir: str, company_name: str) -> None:
        save_dir = f"{weights_dir}/{company_name}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{save_dir}/varmax.ts", "wb") as f:
            pickle.dump(self.predictor, f)

    @classmethod
    def from_weights(cls, weights_dir: str, company_name: str) -> "ZooVARMAX":
        save_dir = f"{weights_dir}/{company_name}"
        with open(f"{save_dir}/varmax.ts", "rb") as f:
            return cls(predictor=pickle.load(f))


class ZooProphet(ZooModel):
    def __init__(self, predictor = None):
        self.predictor = predictor

    def train(self, data: pd.DataFrame) -> None:
        self.predictor = Prophet(
            freq="D",
            n_changepoints=int(len(data) / 12),
            add_country_holidays={"country_name": "USA"},
            yearly_seasonality=True
        )
        self.predictor.fit(y=data)

    def predict(self, days: int) -> pd.Series:
        return self.predictor.predict(fh=list(range(1, days+1)))["Close"]

    def save(self, weights_dir: str, company_name: str) -> None:
        save_dir = f"{weights_dir}/{company_name}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{save_dir}/prophet.ts", "wb") as f:
            pickle.dump(self.predictor, f)

    @classmethod
    def from_weights(cls, weights_dir: str, company_name: str) -> "ZooProphet":
        save_dir = f"{weights_dir}/{company_name}"
        with open(f"{save_dir}/prophet.ts", "rb") as f:
            return cls(predictor=pickle.load(f))


class ZooETS(ZooModel):
    def __init__(self, predictor = None):
        self.predictor = predictor

    def train(self, data: pd.DataFrame) -> None:
        self.predictor = AutoETS(auto=True)
        self.predictor.fit(y=data)

    def predict(self, days: int) -> pd.Series:
        return self.predictor.predict(fh=list(range(1, days+1)))["Close"]

    def save(self, weights_dir: str, company_name: str) -> None:
        save_dir = f"{weights_dir}/{company_name}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{save_dir}/ets.ts", "wb") as f:
            pickle.dump(self.predictor, f)

    @classmethod
    def from_weights(cls, weights_dir: str, company_name: str) -> "ZooETS":
        save_dir = f"{weights_dir}/{company_name}"
        with open(f"{save_dir}/ets.ts", "rb") as f:
            return cls(predictor=pickle.load(f))


class ZooSARIMAX(ZooModel):
    def __init__(self, predictor = None):
        self.predictor = predictor

    def train(self, data: pd.DataFrame) -> None:
        self.predictor = SARIMAX(order=(5, 0, 5))
        self.predictor.fit(y=data)

    def predict(self, days: int) -> pd.Series:
        return self.predictor.predict(fh=list(range(1, days+1)))["Close"]

    def save(self, weights_dir: str, company_name: str) -> None:
        save_dir = f"{weights_dir}/{company_name}"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{save_dir}/sarimax.ts", "wb") as f:
            pickle.dump(self.predictor, f)

    @classmethod
    def from_weights(cls, weights_dir: str, company_name: str) -> "ZooSARIMAX":
        save_dir = f"{weights_dir}/{company_name}"
        with open(f"{save_dir}/sarimax.ts", "rb") as f:
            return cls(predictor=pickle.load(f))
