import os
import sys

sys.path.append(os.getenv('AIRFLOW_HOME'))

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import pipeline

default_args = {
    'owner': 'airflow',
    'start_date': datetime.now(),
    'catchup': False
}

cfg = {
    "donot_pickle": "True"
}

dag = DAG(
    'maintain_pipeline',
    default_args=default_args,
    description='For Auto Maintain',
    schedule=timedelta(days=1)
)

# 创建任务
crawl_sh_hotel = PythonOperator(
    task_id='crawl_sh_hotel',
    python_callable=pipeline.crawl_sh_hotel,
    dag=dag,
    provide_context=True,
    execution_timeout=timedelta(minutes=20),
    executor_config=cfg
)

crawl_sh_visitors = PythonOperator(
    task_id='crawl_sh_visitors',
    python_callable=pipeline.crawl_sh_visitors,
    dag=dag,
    execution_timeout=timedelta(minutes=20),
    executor_config=cfg

)

crawl_hk_visitors = PythonOperator(
    task_id='crawl_hk_visitors',
    python_callable=pipeline.crawl_hk_visitors,
    dag=dag,
    execution_timeout=timedelta(minutes=20)
)


def check_crawl_sh_visitors(**kwargs):
    ti = kwargs['ti']
    result = ti.xcom_pull(task_ids='crawl_sh_visitors')
    if result == 1:
        return 'continue_execution'
    else:
        return 'end_execution'


check_crawl_sh_visitors = BranchPythonOperator(
    task_id='check_crawl_sh_visitors',
    python_callable=check_crawl_sh_visitors,
    provide_context=True,
    dag=dag,
)

continue_execution = EmptyOperator(
    task_id='continue_execution',
    dag=dag,
)

end_execution = EmptyOperator(
    task_id='end_execution',
    dag=dag,
)

impute_hk_visitors = PythonOperator(
    task_id='impute_hk_visitors',
    python_callable=pipeline.impute_hk_visitors,
    dag=dag,
    execution_timeout=timedelta(minutes=20)
)

interpolate_sh_visitors = PythonOperator(
    task_id='interpolate_sh_visitors',
    python_callable=pipeline.interpolate_sh_visitors,
    dag=dag,
    execution_timeout=timedelta(minutes=20)
)

train_sh_visitors_model = PythonOperator(
    task_id='train_sh_visitors',
    python_callable=pipeline.train_sh_visitors,
    dag=dag,
    execution_timeout=timedelta(minutes=20)
)

predict_sh_visitors_model = PythonOperator(
    task_id='predict_sh_visitors',
    python_callable=pipeline.predict_sh_visitors,
    dag=dag,
    execution_timeout=timedelta(minutes=20)
)

train_predict_sh_hotels = PythonOperator(
    task_id='train_predict_sh_hotel',
    python_callable=pipeline.train_predict_sh_hotels,
    dag=dag,
    execution_timeout=timedelta(minutes=20)
)

# 设置依赖关系
# 函数 1, 2, 3 可以并行
# 设置依赖关系
[crawl_sh_visitors, crawl_hk_visitors] >> check_crawl_sh_visitors
check_crawl_sh_visitors >> continue_execution >> crawl_sh_hotel
check_crawl_sh_visitors >> end_execution

continue_execution >> impute_hk_visitors >> interpolate_sh_visitors >> train_sh_visitors_model >> predict_sh_visitors_model >> end_execution
continue_execution >> crawl_sh_hotel >> train_predict_sh_hotels >> end_execution
