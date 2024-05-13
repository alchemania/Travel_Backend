import logging
import pickle
import random
import re
import time
import uuid
from datetime import date
from io import StringIO
from typing import List, Dict, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from lxml.html.diff import parse_html
from pandas import DataFrame

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
        new_urls = [url for url in current_urls if url not in existing_urls]

        for url in new_urls:
            task_id = self.identifier + '_' + str(uuid.uuid5(uuid.NAMESPACE_DNS, url)).replace('-', '')[:8]
            self.tasks[task_id] = url

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
    identifier = 'visitor_spider'
    pd_columns = ["date", "foreign_entry", "hkmo_entry", "tw_entry"]
    locs = ["外国人", "港澳同胞", "台湾同胞"]
    prefix = 'https://tjj.sh.gov.cn'
    start_url = 'https://tjj.sh.gov.cn/ydsj56/index.html'
    template = 'https://tjj.sh.gov.cn/ydsj56/index_{page}.html'

    def __init__(self, spider_status: str, tasks: Dict[str, str] = None):
        super().__init__(spider_status, tasks)


class ShHotelSpider(ShStatsSpider):
    identifier = 'sh_hotel_spd'
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

    def download_and_load_csv(self):
        headers = {
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7'
        }
        response = requests.get(self.data_url, headers=headers)
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


# unit test
if __name__ == '__main__':
    spider = ShHotelSpider('new')
    spider.run()
    print(spider.data)
    spider = HKVisitorsSpider()
    spider.run()
    print(spider.data)
