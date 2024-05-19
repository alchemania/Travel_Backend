import os
import sys

sys.path.append(os.getenv('AIRFLOW_HOME'))

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pipeline

default_args = {
    'owner': 'airflow',
    'start_date': datetime.now() - timedelta(days=1),
    'catchup': False
}

dag = DAG(
    'model_pipeline',
    default_args=default_args,
    description='For Auto Maintain',
)

train = PythonOperator(
    task_id='train_sh_visitors',
    python_callable=pipeline.train_sh_visitors,
    dag=dag,
)

predict = PythonOperator(
    task_id='predict_sh_visitors',
    python_callable=pipeline.predict_sh_visitors,
    dag=dag,
)

tp = PythonOperator(
    task_id='train_predict_sh_hotels',
    python_callable=pipeline.train_predict_sh_hotels,
    dag=dag,
)

[train, tp] >> predict
