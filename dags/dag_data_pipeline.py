import os
import sys

sys.path.append(os.getenv('AIRFLOW_HOME'))

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pipeline

default_args = {
    'owner': 'airflow',
    'start_date': datetime.now(),
    'catchup': False,
    'schedule_interval': None
}

dag = DAG(
    'data_pipeline',
    default_args=default_args,
    description='For Auto Maintain',
)

# 创建任务
crawl_sh_hotel = PythonOperator(
    task_id='crawl_sh_hotel',
    python_callable=pipeline.crawl_sh_hotel,
    dag=dag,
)

crawl_sh_visitors = PythonOperator(
    task_id='crawl_sh_visitors',
    python_callable=pipeline.crawl_sh_visitors,
    dag=dag,
)

crawl_hk_visitors = PythonOperator(
    task_id='crawl_hk_visitors',
    python_callable=pipeline.crawl_hk_visitors,
    dag=dag,
)

# 设置依赖关系
# 函数 1, 2, 3 可以并行
[crawl_sh_visitors, crawl_sh_hotel] >> crawl_hk_visitors
