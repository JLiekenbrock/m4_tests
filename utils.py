from transformers import AutoModelForCausalLM
import pandas as pd
import torch
from chronos import BaseChronosPipeline
from datasetsforecast.m4 import M4
import numpy as np
np.random.seed(42)
from multiprocessing import Pool, cpu_count, Manager
import logging
logging.getLogger("prophet.plot").disabled = True
from prophet import Prophet
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

def prepare_data(sample_size=1000,series_cutoff=48, min_series_length=42):

    path = 'data/m4/datasets'
    group = 'Monthly'

    def read_and_melt(file):
        df = pd.read_csv(file)
        df.columns = ['unique_id'] + list(range(1, df.shape[1]))
        df = pd.melt(df, id_vars=['unique_id'], var_name='row', value_name='y')
        df = df.dropna()

        return df

    df_train = read_and_melt(file=f'{path}/{group}-train.csv')
    df_test = read_and_melt(file=f'{path}/{group}-test.csv')

    len_train = df_train.groupby('unique_id').agg({'row': 'max'}).reset_index()

    len_train.columns = ['unique_id', 'len_serie']
    df_test = df_test.merge(len_train, on=['unique_id'])
    df_test['row'] = df_test['row'] + df_test['len_serie']
    df_test.drop('len_serie', axis=1, inplace=True)

    df = pd.concat([df_train, df_test])
    df = df.sort_values(['unique_id', 'row']).reset_index(drop=True)
    
    groups = np.random.choice(df["unique_id"].unique(), size=sample_size, replace=False)

    df = df.set_index(["unique_id","row"]).sort_index().groupby(level=0).tail(series_cutoff)

    df = df.loc[df.index.get_level_values('unique_id').isin(groups)].reset_index()

    dates = pd.read_csv("data/m4/datasets/M4-info.csv")
    dates = (
        dates[ (dates["SP"]=="Monthly") & (dates["M4id"].isin(groups)) & (dates["StartingDate"].str.len()==14)]
            .assign(StartingDate = lambda x : pd.to_datetime(x["StartingDate"]))
            .loc[lambda df: df["StartingDate"]<'2020-01-01']
            .rename(columns = {"M4id":"unique_id"})
    )

    dates = (
        df.groupby("unique_id", as_index=False)["row"].max()
        .merge(dates[["unique_id", "StartingDate"]], on="unique_id")
        .assign(
            ds=lambda x: x.apply(
                lambda row: pd.date_range(start=row["StartingDate"], periods=row["row"], freq='ME'),
                axis=1
            )
        ) 
        .explode("ds")  
        .assign(
            row = lambda x : x.groupby("unique_id")["StartingDate"].rank(method="first", ascending=True).astype(int),
            ds = lambda x : pd.to_datetime(x["ds"])
        )
    )

    df["max"] = df.groupby("unique_id")["row"].transform(max)

    df = df.merge(dates,on=["unique_id","row"])

    df = df[df["max"]>min_series_length]

    df["test"] = df["row"] > df["max"] -3

    train = df[df["test"]==0].sort_values(by=["unique_id","ds"]).drop(columns=["test","max","row","StartingDate"])
    test = df[df["test"]==1].sort_values(by=["unique_id","ds"]).drop(columns=["test","max","row","StartingDate"])

    train.to_csv("train.csv",index=False)
    test.to_csv("test.csv",index=False)

    return train, test

def wape(
    df:  pd.DataFrame,
    models: list[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) ->  pd.DataFrame:
    """Weighted Absolute Percentage Error (WAPE)

    WAPE measures the relative prediction
    accuracy of a forecasting method by calculating the percentual deviation
    of the prediction and the observed value at a given time and
    averages these devations over the length of the series.
    The closer to zero an observed value is, the higher penalty WAPE loss
    assigns to the corresponding error."""
    if isinstance(df, pd.DataFrame):
        res = (
            df[models]
            .sub(df[target_col], axis=0)
            .abs()
            .groupby(df[id_col], observed=True)
            .sum()
            .div(
                 (df[models]
                .abs()
                .groupby(df[id_col], observed=True).sum()), axis=0)
            )
        res.index.name = id_col
        res = res.reset_index()
    return res

class ProphetForecast:
    def __init__(self, data, **args):
        self.args = args 
        self.models = {}

        for index in data["unique_id"].unique():
            self.models[index] = Prophet(**self.args)

    def _prepare_data(self, data_subset):
        data_subset = data_subset.drop(columns='unique_id', axis=1)
        data_subset = data_subset.rename(columns={'date': 'ds', 'target': 'y'})
        return data_subset

    def _fit_single_model(self, unique_id, data_subset):
        model = Prophet(**self.args)
        data_subset = self._prepare_data(data_subset)
        model.fit(data_subset)
        return unique_id, model  # Return model for later prediction

    def fit(self, df):
        data_groups = [df[df["unique_id"] == unique_id] for unique_id in self.models.keys()]
        
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_model = {executor.submit(self._fit_single_model, unique_id, data_group): unique_id 
                               for unique_id, data_group in zip(self.models.keys(), data_groups)}
            
            for future in as_completed(future_to_model):
                unique_id, model = future.result()
                self.models[unique_id] = model

    def _predict_single_model(self, unique_id, h):
        model = self.models[unique_id]
        future = model.make_future_dataframe(periods=h, include_history=False, freq='ME')
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']]

    def format_prediction(self, forecasts, test):
        model_name = str(type(self).__name__)
        return (
            pd.concat([test[["unique_id", "y"]].reset_index(), pd.concat(forecasts).reset_index()], axis=1)
                .drop(columns="index")
                .rename(columns={"yhat": model_name , "yhat_upper": f"{model_name}-hi-95", "yhat_lower": f"{model_name}-lo-95"})
                [["unique_id", "ds", model_name, f"{model_name}-hi-95", f"{model_name}-lo-95"]]
        )

    def predict(self, test, h):
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_model = {executor.submit(self._predict_single_model, unique_id, h): unique_id
                               for unique_id in self.models.keys()}
            
            forecasts = []
            for future in as_completed(future_to_model):
                forecasts.append(future.result())
        
        return self.format_prediction(forecasts, test)

class LLM:
    def __init__(self, input_length, device):
        """
        Base class for LLM-based predictors.
        """
        self.input_length = input_length
        self.device = device
        self.pred_proba = None
        self.trained_ = False

    def preprocess_data(self, train):
        """
        Preprocess data for input to the model.
        To be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def predict(self, train, test, h):
        """
        Generate predictions.
        To be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    

    def format_prediction(self, prediction, test):
        return (
            pd.concat( [
                            test[["unique_id","ds"]].reset_index(drop=True), 
                            prediction.reset_index(drop=True)
                        ]
                        ,axis=1
                )
        )

    def format_input(self, train, input_length):
        return (
            train
                    .sort_values(by=["unique_id","ds"])
                    .groupby("unique_id")
                    .tail(input_length)
                    .groupby("unique_id")
                    .agg({'y': lambda x: x.tolist()})
        )    

class ChronosPredictor(LLM):
    def __init__(self, input_length, device):
        super().__init__(input_length, device)
        
        self.model = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base",
            device_map=device, 
            torch_dtype=torch.float32,
        )

    def preprocess_data(self, train):
        vals = self.format_input(train, self.input_length)
        return [torch.tensor(val) for val in vals["y"]]

    def predict(self, train, test, h):

        model_name = str(type(self).__name__)

        quantiles = [0.025,0.975]

        data = self.preprocess_data(train)
        quantiles, mean = self.model.predict_quantiles(context=data, prediction_length=h, quantile_levels=quantiles)

        fc = (quantiles.numpy().flatten().reshape(len(mean.numpy().flatten()),len(quantiles)))

        fc = pd.DataFrame({
            model_name: mean.numpy().flatten(),
            f"{model_name}-lo-95": fc[:,0].flatten(), 
            f"{model_name}-hi-95": fc[:,1].flatten() 
        })
        
        return self.format_prediction(fc, test)
    
    def __sklearn_is_fitted__(self):
        if self.trained_:
            return True
        else:
            return False


class TimeMoEPredictor(LLM):
    def __init__(self, input_length, device):
        super().__init__(input_length, device)
        self.model = AutoModelForCausalLM.from_pretrained(
            'Maple728/TimeMoE-50M',
            device_map=device,
            trust_remote_code=True
        ).to(self.device)

    def get_params(self,deep=True):
        return (
            {
                "input_length": self.input_length,
                "device": self.device
            }
        )
    def preprocess_data(self, train):
        vals = self.format_input(train, self.input_length)
        values_tensor = torch.tensor(vals["y"].tolist())
        mean = values_tensor.mean(dim=-1, keepdim=True)
        std = values_tensor.std(dim=-1, keepdim=True)
        normed_seqs = (values_tensor - mean) / std
        return normed_seqs, mean, std

    def fit(self):
        print("already fitted")
        self.trained_ = True

    def predict(self, X, test, h):

        model_name = str(type(self).__name__)
        
        normed_seqs, mean, std = self.preprocess_data(X)
        output = self.model.generate(normed_seqs, max_new_tokens=h)
        normed_predictions = output[:, -h:].to('cpu')
        predictions =  pd.DataFrame({model_name: (normed_predictions * std + mean).numpy().flatten()})

        return self.format_prediction(predictions, test)

    def __sklearn_is_fitted__(self):
        if self.trained_:
            return True
        else:
            return False
