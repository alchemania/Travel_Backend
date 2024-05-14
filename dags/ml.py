import sys
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
    'machine_learning',
    default_args=default_args,
    description='For Auto Maintain',
    schedule=timedelta(days=1)
)

train = PythonOperator(
    task_id='train',
    python_callable=pipeline.train_sh_visitors,
    dag=dag,
)

predict = PythonOperator(
    task_id='predict',
    python_callable=pipeline.predict_sh_visitors,
    dag=dag,
)

train >> predict
