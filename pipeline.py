# python packages
import os
from datetime import datetime, date
from pathlib import Path
import logging
import pickle
import random
import re
import time
import uuid
from io import StringIO
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from django.utils.dateparse import parse_datetime
from lxml.html.diff import parse_html
from pandas import DataFrame
import django
import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from pypots.imputation import Transformer
from pypots.optim import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'TravelServer.settings')
django.setup()

from django_pandas.io import read_frame

logger = logging.getLogger(__name__)

# django imports
from api.models import *
from django.db.models import Max
from TravelServer import settings

# Setup logging
logging.basicConfig(level=logging.INFO)

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36",
]


def request_with_retry(url: str, max_retries: int = 3) -> Optional[requests.Response]:
    session = requests.Session()
    headers = {'User-Agent': random.choice(user_agents)}
    session.headers.update(headers)

    for attempt in range(max_retries):
        try:
            response = session.get(url)
            if response.status_code == 200:
                response.encoding = response.apparent_encoding
                return response
            else:
                logging.info(f"Non-200 status code: {response.status_code}")
        except requests.RequestException as e:
            logging.error(f"Request failed: {e}")

        time.sleep(random.uniform(1, 1.5))
        logging.info(f"Retry {attempt + 1}/{max_retries}")

    logging.error("All retries failed.")
    return None


def get_pure_text(html_content: str) -> List[str]:
    tree = parse_html(html_content)
    texts = tree.xpath('.//text()')
    return [text.strip() for text in texts if text.strip()]


class ShStatsSpider:
    identifier = ''
    pd_columns = []
    locs = []
    prefix = ''
    start_url = ''
    template = ''

    def __init__(self, spider_status: str, tasks: Dict[str, str] = None):
        self.spider_status = spider_status
        self.tasks = tasks or {}
        self.task_pointer = 0
        self.skipped_tasks = []
        self.data = []

    def _get_tasks(self):
        response = request_with_retry(self.start_url)
        if not response:
            logging.error("Failed to fetch start page.")
            return

        page_count = self.extract_page_count(response.text)
        if self.spider_status == 'new':
            self.fetch_all_tasks(page_count)
        elif self.spider_status == 'update':
            self.update_tasks(response.text)

    def extract_page_count(self, html_text: str) -> int:
        match = re.search(r'totalPage:\s*(\d+)', html_text)
        if not match:
            logging.warning("No page count found, using default of 8")
            return 8
        return int(match.group(1))

    def fetch_all_tasks(self, page_count: int):
        for i in range(1, page_count + 1):
            url = self.start_url if i == 1 else self.template.format(page=i)
            response = request_with_retry(url)
            if response:
                urls = self.extract_urls(response.text)
                for url in urls:
                    task_id = self.identifier + '_' + str(uuid.uuid5(uuid.NAMESPACE_DNS, url)).replace('-', '')[:8]
                    self.tasks[task_id] = url
                    self.spider_status = 'running'

    def update_tasks(self, current_page_html: str):
        """
        Updates the list of tasks by adding only new URLs that were not present in the previous tasks.
        """
        current_urls = self.extract_urls(current_page_html)
        existing_urls = set(self.tasks.values())
        new_urls = list(set(current_urls).difference(existing_urls))
        self.tasks = {
            self.identifier + '_' + str(uuid.uuid5(uuid.NAMESPACE_DNS, url)).replace('-', '')[:8]: url
            for url in new_urls
        }
        logging.info(f"Added {len(new_urls)} new tasks.")
        self.spider_status = 'running'

    def extract_urls(self, html_text: str) -> List[str]:
        soup = BeautifulSoup(html_text, 'html.parser')
        ul_tag = soup.find('ul')
        return [self.prefix + a['href'] for a in ul_tag.find_all('a', href=True) if a and 'href' in a.attrs]

    def run(self):
        if self.spider_status == 'stopped':
            logging.error('Stopped spider cannot use run, please use spider.resume instead.')
            return
        try:
            self._get_tasks()
            while self.task_pointer < len(self.tasks):
                task_url = list(self.tasks.values())[self.task_pointer]
                response = request_with_retry(task_url)
                if response:
                    data = self.process_page(response.text)
                    self.data.append(data)
                    self.task_pointer += 1
                else:
                    logging.error("Failed to process task due to request failure.")
        except Exception as e:
            self.dump()
            logging.error("Error in get tasks, Spider Dumped:", e)

    def process_page(self, html_content: str) -> List:
        texts = get_pure_text(html_content)
        date_info = self.extract_date(texts)
        entries = [self.extract_entry(loc, texts) for loc in self.locs]
        return [date_info] + entries

    def extract_date(self, texts: List[str]) -> date:
        for text in texts:
            match = re.search(r'(\d{4})年(\d{1,2})月', text)
            if match:
                return date(year=int(match.group(1)), month=int(match.group(2)), day=1)
        logging.warning("Date not found in texts.")
        return None

    def extract_entry(self, location: str, texts: List[str]) -> Optional[float]:
        for i, text in enumerate(texts):
            if location in text:
                for j in range(i + 1, len(texts)):
                    if re.match(r'^-?\d+(\.\d+)?$', texts[j]):
                        return float(texts[j])
        logging.warning(f"Entry for {location} not found.")
        return None

    def data_frame(self) -> DataFrame:
        return DataFrame(self.data, columns=self.pd_columns)

    def dump(self):
        self.spider_status = 'stopped'
        with open(f"{self.identifier}_dumped.pkl", 'wb') as file:
            pickle.dump(self, file)

    def resume(self):
        """
        Resumes the spider from a stopped state, reloading its state from a saved dump if necessary.
        """
        if self.spider_status != 'stopped':
            logging.error('Resume operation is only valid if the spider is in stopped status.')
            return

        # Optional: Load the spider's state from a file if not already loaded
        self.load_state()

        self.spider_status = 'running'
        logging.info('Resuming spider operation...')
        self.run()

    @staticmethod
    def load_state(path):
        """
        Loads the spider's state from a pickle file if the in-memory data is not present or incomplete.
        """
        try:
            with open(path, 'rb') as file:
                loaded_spider = pickle.load(file)
            logging.info('Spider state successfully loaded from dump file.')
            return loaded_spider
        except FileNotFoundError:
            logging.error('No dump file found to resume spider operation.')
        except Exception as e:
            logging.error(f'Failed to load spider state: {str(e)}')


class ShVisitorsSpider(ShStatsSpider):
    # rewrite
    identifier = 'visitors_spider'
    pd_columns = ["date", "foreign_entry", "hkmo_entry", "tw_entry"]
    locs = ["外国人", "港澳同胞", "台湾同胞"]
    prefix = 'https://tjj.sh.gov.cn'
    start_url = 'https://tjj.sh.gov.cn/ydsj56/index.html'
    template = 'https://tjj.sh.gov.cn/ydsj56/index_{page}.html'

    def __init__(self, spider_status: str, tasks: Dict[str, str] = None):
        super().__init__(spider_status, tasks)


# test passed
class ShHotelSpider(ShStatsSpider):
    identifier = 'hotel_spider'
    pd_columns = ["日期", "平均出租率", "五星级平均出租率", "平均房价", "五星级平均房价", "平均房价增长",
                  "五星级房价增长"]
    locs = ["星级饭店客房平均出租率", "五星级", "星级饭店平均房价", "五星级", "星级饭店平均房价增长", "五星级"]
    prefix = 'https://tjj.sh.gov.cn'
    start_url = 'https://tjj.sh.gov.cn/ydsj57/index.html'
    template = 'https://tjj.sh.gov.cn/ydsj57/index_{page}.html'

    def __init__(self, spider_status: str, tasks: Dict[str, str] = None):
        super().__init__(spider_status, tasks)


class HKVisitorsSpider:
    def __init__(self):
        self.tasks = [
            'https://www.immd.gov.hk/opendata/hkt/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv'
        ]
        self.data = None

    def download_and_load_csv(self, url):
        headers = {
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7'
        }
        response = requests.get(url, headers=headers)
        response.encoding = response.apparent_encoding
        csv_data = StringIO(response.text)
        return pd.read_csv(csv_data)

    def consume(self, url):
        df = self.download_and_load_csv(url)
        # 使用pd.to_datetime确保日期格式正确
        df['日期'] = pd.to_datetime(df['日期'], format='%d-%m-%Y')

        # 计算每个指定的聚合
        aggregations = {
            'HK_airport_entry': pd.NamedAgg(column='香港居民', aggfunc=lambda x: x[
                (df['入境 / 出境'] == '入境') & (df['管制站'] == '機場')].sum()),
            'CN_airport_entry': pd.NamedAgg(column='內地訪客', aggfunc=lambda x: x[
                (df['入境 / 出境'] == '入境') & (df['管制站'] == '機場')].sum()),
            'global_airport_entry': pd.NamedAgg(column='其他訪客',
                                                aggfunc=lambda x: x[
                                                    (df['入境 / 出境'] == '入境') & (df['管制站'] == '機場')].sum()),
            'airport_entry': pd.NamedAgg(column='總計', aggfunc=lambda x: x[
                (df['入境 / 出境'] == '入境') & (df['管制站'] == '機場')].sum()),
            'HK_airport_departure': pd.NamedAgg(column='香港居民', aggfunc=lambda x: x[
                (df['入境 / 出境'] == '出境') & (df['管制站'] == '機場')].sum()),
            'CN_airport_departure': pd.NamedAgg(column='內地訪客', aggfunc=lambda x: x[
                (df['入境 / 出境'] == '出境') & (df['管制站'] == '機場')].sum()),
            'global_airport_departure': pd.NamedAgg(column='其他訪客',
                                                    aggfunc=lambda x: x[(df['入境 / 出境'] == '出境') & (
                                                            df['管制站'] == '機場')].sum()),
            'airport_departure': pd.NamedAgg(column='總計', aggfunc=lambda x: x[
                (df['入境 / 出境'] == '出境') & (df['管制站'] == '機場')].sum()),
        }

        # 进行分组和聚合
        grouped = df.groupby(df['日期'].dt.date).agg(**aggregations).reset_index()
        grouped.rename(columns={'日期': 'date'}, inplace=True)
        # 打印结果
        return grouped

    def run(self):
        for url in self.tasks:
            self.data = self.consume(url)

    def data_frame(self):
        return self.data


# test passed
def crawl_sh_hotel(*args, **kwargs):
    logger.info("Starting the hotel spider task")
    tasks = DbSpider.objects.filter(unique_id__contains=ShHotelSpider.identifier).all()
    tasks = read_frame(tasks).to_dict(orient='list')
    tasks = dict(zip(tasks['unique_id'], tasks['url']))
    spider = ShHotelSpider('update', tasks=tasks)
    spider.run()
    logger.info("Hotel spider task fetching completed successfully")

    if spider.tasks:
        logger.info("Hotel spider new data fetched")
        objs = [DbSpider(unique_id=uid, url=url) for uid, url in spider.tasks.items()]
        DbSpider.objects.bulk_create(objs, ignore_conflicts=True)
        logger.info("New spider tasks added to database")

        data = spider.data_frame()
        instances = [
            DbShHotel(
                DATE=row[ShHotelSpider.pd_columns[0]],
                avg_rent_rate=row[ShHotelSpider.pd_columns[1]],
                avg_rent_rate_5=row[ShHotelSpider.pd_columns[2]],
                avg_price=row[ShHotelSpider.pd_columns[3]],
                avg_price_5=row[ShHotelSpider.pd_columns[4]],
            )
            for index, row in data.iterrows()
        ]
        DbShHotel.objects.bulk_create(instances, ignore_conflicts=True)
        logger.info("Hotel data inserted into database")
        return 1
    else:
        logger.info("Hotel spider task no new data fetched")
        return 0


def crawl_sh_visitors(*args, **kwargs):
    logger.info("Starting the sh visitor spider task")
    tasks = DbSpider.objects.filter(unique_id__contains=ShVisitorsSpider.identifier).all()
    tasks = read_frame(tasks).to_dict(orient='list')
    tasks = dict(zip(tasks['unique_id'], tasks['url']))
    spider = ShVisitorsSpider('update', tasks=tasks)
    spider.run()
    logger.info("sh visitors spider task fetching completed successfully")

    if spider.tasks:
        logger.info("sh visitors spider new data fetched")
        objs = [DbSpider(unique_id=uid, url=url) for uid, url in spider.tasks.items()]
        DbSpider.objects.bulk_create(objs, ignore_conflicts=True)
        logger.info("New spider tasks added to database")

        data = spider.data_frame()
        instances = [
            DbShvisitorsMonthly(
                DATE=row[ShVisitorsSpider.pd_columns[0]],
                FOREIGN=int(row[ShVisitorsSpider.pd_columns[1]]) * 10000,
                HM=int(row[ShVisitorsSpider.pd_columns[2]]) * 10000,
                TW=int(row[ShVisitorsSpider.pd_columns[3]]) * 10000,
            )
            for index, row in data.iterrows()
        ]
        DbShvisitorsMonthly.objects.bulk_create(instances, ignore_conflicts=True)
        logger.info("SH Visitor data inserted into database")
        return 1
    else:
        logger.info("SH Visitor spider task no new data fetched")
        return 0


# test passed
def crawl_hk_visitors(*args, **kwargs):
    logger.info("Starting the HK visitors spider task")
    spider = HKVisitorsSpider()
    spider.run()
    logger.info("HK visitors spider task fetching completed successfully")
    max_date_db = DbHkVisitors.objects.aggregate(max_date=Max('date'))['max_date'].date()
    max_date_spd = spider.data['date'].max()
    if max_date_spd > max_date_db or max_date_db is None:
        logger.info("HK visitors spider new data fetched")
        data = spider.data[spider.data['date'] > max_date_db]
        instances = [
            DbHkVisitors(
                date=row['date'],
                HK_airport_entry=row['HK_airport_entry'],
                CN_airport_entry=row['CN_airport_entry'],
                global_airport_entry=row['global_airport_entry'],
                airport_entry=row['airport_entry'],
                HK_airport_departure=row['HK_airport_departure'],
                CN_airport_departure=row['CN_airport_departure'],
                global_airport_departure=row['global_airport_departure'],
                airport_departure=row['airport_departure']
            )
            for index, row in data.iterrows()
        ]
        DbHkVisitors.objects.bulk_create(instances, ignore_conflicts=True)
        logger.info("HK visitors data updated in database")
        return 1
    else:
        logger.info("HK visitors spider no data fetched")
        return 0


# test passed
def impute_hk_visitors(*args, **kwargs):
    # fetch Data
    scaler = MinMaxScaler()
    df_ori = read_frame(DbHkVisitors.objects.all(), index_col='date', datetime_index=True)

    # MinMax for each row
    df = pd.DataFrame(
        data=scaler.fit_transform(df_ori.values),
        columns=df_ori.columns,
        index=df_ori.index
    )

    # Reindex
    index = pd.date_range(start='2019-01-01', end=df_ori.index.max(), freq='D')
    df_ori = df_ori.reindex(index)
    df = df.reindex(index)

    # sample&features
    num_samples = df.values.shape[0]
    num_features = df.values.shape[1]  # 减去date

    transformer = Transformer(
        n_steps=num_samples,
        n_features=num_features,
        n_layers=4,
        d_model=192,
        d_ffn=128,
        n_heads=6,
        d_k=32,
        d_v=64,
        dropout=0.1,
        attn_dropout=0,
        epochs=400,
        optimizer=Adam(lr=1e-3),
        device=torch.device(
            "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
        patience=50,
    )
    ds = {"X": df.values.reshape(1, num_samples, -1)}
    transformer.fit(ds)

    # impute and convert back
    impute = pd.DataFrame(
        scaler.inverse_transform(transformer.impute(ds).squeeze()),
        columns=df.columns,
        index=df.index)
    # impute = impute.apply(lambda col: scalers[col.name].inverse_transform(col.values.reshape(-1, 1)).flatten())
    impute = impute.astype('int')
    impute = pd.DataFrame({
        col: np.where(
            (np.isnan(df.values) ^ np.isnan(impute.values)).reshape(num_samples, num_features)[:, 0],
            impute[col],
            df_ori[col]
        ) for col
        in
        df_ori.columns
    }, index=df.index)

    # 操作数据库
    instances = [
        DbHkVisitorsImputed(
            date=index,
            HK_airport_entry=row['HK_airport_entry'],
            CN_airport_entry=row['CN_airport_entry'],
            global_airport_entry=row['global_airport_entry'],
            airport_entry=row['airport_entry'],
            HK_airport_departure=row['HK_airport_departure'],
            CN_airport_departure=row['CN_airport_departure'],
            global_airport_departure=row['global_airport_departure'],
            airport_departure=row['airport_departure']
        )
        for index, row in impute.iterrows()
    ]
    DbHkVisitorsImputed.objects.bulk_create(instances, update_conflicts=True, update_fields=[
        'HK_airport_entry', 'CN_airport_entry', 'global_airport_entry', 'airport_entry',
        'HK_airport_departure', 'CN_airport_departure', 'global_airport_departure', 'airport_departure'
    ])
    logger.info("HK visitors data updated in database")


# test passed
def interpolation(sh_ori, hk_ori, noise: float = 0.1):
    regular_slice_start = '2011-01-01'
    regular_slice_end = '2019-12-31'
    special_slice_start = '2020-01-01'
    hk_regular_slice = hk_ori[:regular_slice_end]
    hk_special_slice = hk_ori[special_slice_start:]
    frames = []  # 存储每年DataFrame的列表

    # 生成到最后一个月最后一天的完整日期范围，避免最后一个月没有值
    full_date_range = pd.date_range(start=sh_ori.index.min(),
                                    end=sh_ori.index.max().to_period('M').to_timestamp('M'), freq='D')

    # 重采样并前向填充
    sh_daily_fake = sh_ori.reindex(full_date_range).ffill()

    # hk only 2019 -> 2011~2019
    for year in range(date.fromisoformat(regular_slice_start).year,
                      date.fromisoformat(regular_slice_end).year + 1):
        # 复制2019年数据，更新年份
        df_temp = hk_regular_slice.copy()
        df_temp.index = df_temp.index.map(lambda x: x.replace(year=year))

        # 处理闰年，添加2月29日
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):  # 判断闰年
            feb_28 = df_temp.loc[f'{year}-02-28']
            df_temp.loc[pd.to_datetime(f'{year}-02-29')] = feb_28.values
            # # 插入2月29日，比例与前一天相同
            # df_temp = pd.concat([
            #     df_temp[:f'{year}-02-28'],
            #     pd.DataFrame({'global_airport_entry': [feb_28['global_airport_entry']]},
            #                  index=[pd.Timestamp()]), df_temp[f'{year}-02-29':]
            # ])

        frames.append(df_temp)

    # 合并所有年份的DataFrame
    hk_former = pd.concat(frames)
    hk_former = hk_former.apply(pd.to_numeric, errors='coerce')
    std = np.std(hk_former.values)
    noise = np.random.normal(0, std * noise, hk_former.shape)
    hk_former = hk_former + noise
    hk_all = pd.concat([hk_former, hk_special_slice])

    # 计算hk ratio
    reg_ratio = pd.DataFrame(index=hk_all.index, columns=hk_all.columns)
    for column in hk_all.columns:
        # 对于每个月，计算该月每一天的值占该月总和的比例
        reg_ratio[column] = hk_all[column] / hk_all[column].resample('ME').transform('sum')

    sh_daily_interpolated = sh_daily_fake.multiply(reg_ratio['global_airport_entry'].head(len(sh_daily_fake)),
                                                   axis=0).astype('int')
    return sh_daily_interpolated


def melt(df):
    if 'date' not in df.columns:
        df = df.reset_index(names='date')
        df_ori = pd.melt(df, id_vars=['date'], var_name='unique_id', value_name='y')
        df_ori.rename(columns={'date': 'ds'}, inplace=True)
        return df_ori
    else:
        df_ori = pd.melt(df, id_vars=['date'], var_name='unique_id', value_name='y')
        df_ori.rename(columns={'date': 'ds'}, inplace=True)
        return df_ori


# melt and pivot are all implemented by django_pandas


def cut(df, start_date='2020-01-01', end_date='2023-06-01'):
    # 使用布尔索引选择不在指定范围内的数据
    df = df.loc[(df.index < start_date) | (df.index > end_date)].sort_index()
    # 创建一个递减的日期索引
    new_index = pd.date_range(end=df.index.max(), periods=len(df))
    # 将新的日期索引应用到数据框架
    df.set_index(new_index, inplace=True)
    # 使用布尔索引选择不在指定范围内的数据
    return df


# test passed
def interpolate_sh_visitors(*args, **kwargs):
    sh = DbShvisitorsMonthly.objects.all()
    sh = read_frame(sh, index_col='DATE', datetime_index=True, coerce_float=True)
    hk = DbHkVisitorsImputed.objects.all()
    hk = read_frame(hk, index_col='date', datetime_index=True, coerce_float=True)
    interpolate = interpolation(sh_ori=sh, hk_ori=hk, noise=0.05)

    # 操作数据库
    instances = [
        DbShvisitorsDaily(
            DATE=index,
            FOREIGN=row['FOREIGN'],
            HM=row['HM'],
            TW=row['TW'],
        ) for index, row in interpolate.iterrows()
    ]
    rows = DbShvisitorsMonthly.objects.bulk_create(instances, ignore_conflicts=True)
    logger.info(f"Sh visitors Interpolated data updated in database, {len(rows)}")


def _get_latest_model(directory, *args, **kwargs):
    logger.info("Searching for the latest model directory")
    max_num = -1
    max_folder = None
    pattern = re.compile(r'modelgroup_(\d+)')
    try:
        for item in os.listdir(directory):
            full_path = os.path.join(directory, item)
            if os.path.isdir(full_path):
                match = pattern.match(item)
                if match:
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num
                        max_folder = full_path
        logger.info(f"Latest model directory found: {max_folder}")
        return max_folder
    except Exception as e:
        logger.error(f"An error occurred while searching for the latest model directory: {str(e)}")
        return None


def train_sh_visitors(*args, **kwargs):
    logger.info("Starting auto-training task")
    base_dir = os.path.join(settings.BASE_DIR, 'models')
    model_dir = _get_latest_model(base_dir)
    df = DbShvisitorsDaily.objects.all()
    df = read_frame(df, index_col='DATE')
    df = cut(df)
    df = melt(df)
    nf = NeuralForecast.load(model_dir, verbose=False)
    nf.fit(df=df)
    nf.save(str(Path(model_dir).parent.joinpath(f'modelgroup_{datetime.now().strftime("%Y%m%d%H%M%S")}')))
    logger.info("Model training completed successfully")


def predict_sh_visitors(*args, **kwargs):
    logger.info("Starting auto-prediction task")
    try:
        base_dir = os.path.join(settings.BASE_DIR, 'models')
        model_dir = _get_latest_model(base_dir)
        db_data = DbShvisitorsDaily.objects.all()
        dataset = melt(cut(read_frame(db_data, index_col='DATE')))
        iters = 6  # 3*60=180
        nf = NeuralForecast.load(model_dir, verbose=False)
        model_names = ['Auto' + cls.__class__.__name__ for cls in nf.models]
        df_pred = dataset.copy()
        for i in range(iters):
            step = nf.predict(df_pred).reset_index()
            step['y'] = step['AutoPatchTST']
            step = step.drop(columns=model_names)
            df_pred = pd.concat([df_pred, step], axis=0)
        instances = [
            DbShvisitorsDailyPredicted(
                DATE=index,
                FOREIGN=row['global_entry'],
                HM=row['hkmo_entry'],
                TW=row['tw_entry']
            )
            for index, row in df_pred.iterrows()
        ]
        DbHkVisitorsImputed.objects.bulk_create(instances, update_conflicts=True)
        logger.info("Prediction data updated in database")
    except Exception as e:
        logger.error(f"An error occurred in the auto-prediction task: {str(e)}")


def train_predict_sh_hotels(*args, **kwargs):
    df_ori = DbShHotel.objects.all()
    df_ori = read_frame(df_ori, index_col='DATE', datetime_index=True, coerce_float=True)
    scalar = StandardScaler()
    df = pd.DataFrame(scalar.fit_transform(df_ori), index=df_ori.index, columns=df_ori.columns)
    df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max() + relativedelta(years=1), freq='MS'))
    df_ori = df_ori.reindex(
        pd.date_range(start=df_ori.index.min(), end=df_ori.index.max() + relativedelta(years=1), freq='MS'))
    num_samples = df.shape[0]
    num_features = df.shape[1]
    ds = {"X": df.values.reshape(1, num_samples, -1)}  # X for model input
    transformer = Transformer(
        n_steps=num_samples,
        n_features=num_features,
        n_layers=4,
        d_model=128,
        d_ffn=128,
        n_heads=4,
        d_k=32,
        d_v=64,
        dropout=0.05,
        attn_dropout=0,
        epochs=400,
        optimizer=Adam(lr=1e-3),
        device=torch.device(
            'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    )

    transformer.fit(ds)

    # impute and convert back
    impute = pd.DataFrame(
        scalar.inverse_transform(transformer.impute(ds).squeeze()),
        columns=df.columns,
        index=df.index)
    impute = pd.DataFrame({
        col: np.where(
            (np.isnan(df.values) ^ np.isnan(impute.values)).reshape(num_samples, num_features)[:, 0],
            impute[col],
            df_ori[col]
        ) for col
        in
        df_ori.columns
    }, index=df.index)

    # 操作数据库

    instances = [
        DbShHotelPred(
            DATE=index,
            avg_rent_rate=row['avg_rent_rate'],
            avg_rent_rate_5=row['avg_rent_rate_5'],
            avg_price=row['avg_price'],
            avg_price_5=row['avg_price_5'],
        )
        for index, row in impute.iterrows()
    ]
    DbShHotelPred.objects.bulk_create(instances, update_conflicts=True, update_fields=[
        'avg_rent_rate', 'avg_rent_rate_5', 'avg_price', 'avg_price_5'
    ])
    logger.info("SH hotel data updated in database")


if __name__ == '__main__':
    #     pass
    # interpolate_sh_visitors()
    # crawl_sh_visitors()
    # crawl_sh_hotel()
    # crawl_hk_visitors()
    # impute_hk_visitors()
    train_predict_sh_hotels()
    # train_predict_sh_hotels()
