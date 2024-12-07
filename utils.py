from transformers import AutoModelForCausalLM
import pandas as pd
import torch
from chronos import BaseChronosPipeline
from datasetsforecast.m4 import M4
import numpy as np
np.random.seed(42)

def prepare_data(sample_size=1000,series_cutoff=48):
    group = 'Monthly'

    df, *_ = M4.load(directory='data', group=group)
    
    groups = np.random.choice(df["unique_id"].unique(), size=sample_size, replace=False)

    df = df.set_index(["unique_id","ds"]).sort_index().groupby(level=0).tail(series_cutoff)
    df = df.loc[df.index.get_level_values('unique_id').isin(groups)].reset_index()

    dates = pd.read_csv("data/m4/datasets/M4-info.csv")
    dates = (
        dates[ (dates["SP"]=="Monthly") & (dates["M4id"].isin(groups)) & (dates["StartingDate"].str.len()==14)]
            .assign(StartingDate = lambda x : pd.to_datetime(x["StartingDate"]))
            .loc[lambda df: df["StartingDate"]<'2020-01-01']
            .rename(columns = {"M4id":"unique_id"})
    )

    df = (
        df.set_index("unique_id")
            .join(dates[["unique_id","StartingDate"]].set_index("unique_id"))
            .rename(columns={"ds":"row"})
            .sort_values(by=["unique_id","row"])
            .assign(
                ds = lambda x : pd.to_datetime((x["StartingDate"] + pd.to_timedelta(x["row"]*30, unit='D')).dt.date+pd.offsets.MonthEnd(0))
                )
            .dropna(subset="ds")
            .reset_index()
    )

    df["max"] = df.groupby("unique_id")["row"].transform(max)

    df["test"] = df["row"] > df["max"] -3

    train = df[df["test"]==0].sort_values(by=["unique_id","ds"]).drop(columns=["test","row","max","StartingDate"])
    test = df[df["test"]==1].sort_values(by=["unique_id","ds"]).drop(columns=["test","row","max","StartingDate"])

    train.to_csv("train.csv")
    test.to_csv("test.csv")

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

def format_prediction(prediction,test,model_name):
    return (
        pd.concat( [
                        test[["unique_id","ds"]].reset_index(drop=True), 
                        pd.DataFrame(prediction).reset_index(drop=True)
                    ]
                        ,ignore_index=True, axis=1
            )
            .rename(columns={0:"unique_id",1:"ds",2:model_name})

    )

def format_input(train,input_length):
    return (
         train
                .sort_values(by=["unique_id","ds"])
                .groupby("unique_id")
                .tail(input_length)
                .groupby("unique_id")
                .agg({'y': lambda x: x.tolist()})
    )    

import torch
import numpy as np

class LLM:
    def __init__(self, input_length, device):
        """
        Base class for LLM-based predictors.
        """
        self.input_length = input_length
        self.device = device

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

class chronosPredictor(LLM):
    def __init__(self, input_length, device):
        super().__init__(input_length, device)
        self.pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base",
            device_map=device, 
            torch_dtype=torch.float32,
        )

    def preprocess_data(self, train):
        vals = format_input(train, self.input_length)
        return [torch.tensor(val) for val in vals["y"]]

    def predict(self, train, test, h):
        data = self.preprocess_data(train)
        prediction = self.pipeline.predict_quantiles(context=data, prediction_length=h, quantile_levels=[0.5])
        fc = ([el.numpy() for pred in prediction for el in pred])[len(data):]
        fc = np.concatenate(fc)
        print(len(fc))
        return format_prediction(fc, test, str(type(self).__name__))

class TimeMoEPredictor(LLM):
    def __init__(self, input_length, device):
        super().__init__(input_length, device)
        self.model = AutoModelForCausalLM.from_pretrained(
            'Maple728/TimeMoE-50M',
            device_map=device,
            trust_remote_code=True
        ).to(self.device)

    def preprocess_data(self, train):
        vals = format_input(train, self.input_length)
        values_tensor = torch.tensor(vals["y"].tolist())
        mean = values_tensor.mean(dim=-1, keepdim=True)
        std = values_tensor.std(dim=-1, keepdim=True)
        normed_seqs = (values_tensor - mean) / std
        return normed_seqs, mean, std

    def predict(self, train, test, h):
        normed_seqs, mean, std = self.preprocess_data(train)
        output = self.model.generate(normed_seqs, max_new_tokens=h)
        print(len(output))
        normed_predictions = output[:, -h:].to('cpu')
        predictions = normed_predictions * std + mean
        return format_prediction(predictions.numpy().ravel(), test, str(type(self).__name__))
