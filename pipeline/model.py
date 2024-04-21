from datetime import datetime
from pathlib import Path

import pandas as pd
from neuralforecast import NeuralForecast
from typing import Literal


def predict(modelpath, dataset: pd.DataFrame, result_type: str):
    """
    modelpath: dictionary with model parameters
    dataset: pandas dataframe with training and test data, must use melt() 1st
    result_type: either 'full' or 'pred' or 'modelname'
    """
    nf = NeuralForecast.load(modelpath, verbose=False)
    results = []
    iters = 6  # 3*60=180
    model_names = ['Auto' + cls.__class__.__name__ for cls in nf.models]
    if result_type == 'full' or result_type == 'pred':
        for model_name in model_names:
            df_pred = dataset.copy()
            for i in range(iters):
                step = nf.predict(df_pred).reset_index()
                step['y'] = step[model_name]
                step = step.drop(columns=model_names)
                df_pred = pd.concat([df_pred, step], axis=0)
            if result_type == 'full':
                results.append(df_pred)
            elif result_type == 'pred':
                df_pred = pd.concat([df_pred, dataset], axis=0).drop_duplicates(keep=False, ignore_index=True)
                results.append(df_pred)
        return dict(zip(model_names, results))
    elif result_type in model_names:
        df_pred = dataset.copy()
        for i in range(iters):
            step = nf.predict(df_pred).reset_index()
            step['y'] = step[result_type]
            step = step.drop(columns=model_names)
            df_pred = pd.concat([df_pred, step], axis=0)
        return pd.concat([df_pred, dataset], axis=0).drop_duplicates(keep=False, ignore_index=True)


def train(modelpath, data: pd.DataFrame):
    """
    modelpath: dictionary with model parameters
    data: pandas dataframe with training and test data, must use melt() 1st
    """
    nf = NeuralForecast.load(modelpath, verbose=False)
    nf.fit(data)
    nf.save(str(Path(modelpath).parent.joinpath(f'modelgroup_{datetime.now().strftime("%Y%m%d%H%M%S")}')))


# unit test
if __name__ == '__main__':
    path = '../models/modelgroup_20240420172635'
    database_url = "sqlite:///../test.db"
    from sqlalchemy import create_engine

    engine = create_engine(database_url)

    imputed_data_query = f"SELECT * FROM sh_visitors_daily"
    sh_daily = pd.read_sql_query(imputed_data_query, engine, index_col='date', parse_dates=['date'])
    from dataprocess import melt, cut

    p = melt(cut(sh_daily))
    pred = predict(path, p, 'AutoPatchTST')
    train(path, p)
