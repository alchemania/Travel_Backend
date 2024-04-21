import uuid
from datetime import date
import pandas as pd
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
# from tqdm.notebook import tqdm
from tqdm import tqdm
import re
import random
import time
import requests
from lxml import html
from typing import Literal
import pickle

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:57.0) Gecko/20100101 Firefox/57.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/13.0.3 Safari/604.1.38"
]


def request_with_retry(url, max_retries=3, custom_headers={}, ):
    retry_count = 0
    while retry_count < max_retries:
        # 随机选择一个用户代理
        user_agent = random.choice(user_agents)
        headers = {
            'User-Agent': user_agent,
            **custom_headers
        }

        try:
            response = requests.get(url, headers=headers)
            # 检查响应代码，如果不是408，则返回响应
            if response.status_code != 408:
                response.encoding = response.apparent_encoding
                return response
            else:
                print(f"请求超时，尝试次数 {retry_count + 1}/{max_retries}")
        except requests.exceptions.RequestException as e:
            print(f"请求过程中发生错误：{e}")

        retry_count += 1
        time.sleep(random.choice([1, 1.2, 1.5]))  # 在重试之间稍微等待一段时间，可以根据需要调整

    return None  # 所有重试尝试后仍然失败


def get_pure_text(plain_html: str) -> list:
    # 解析HTML
    tree = html.fromstring(plain_html)
    # 提取所有文本节点，并且在每个文本节点之间保持原有的分隔
    texts = tree.xpath('.//text()')
    # 清理每个文本块，移除前后空白符
    cleaned_texts = [text.strip() for text in texts if text.strip()]
    # 清理文本，去除多余的空格和换行
    return cleaned_texts


class BaseSpider:
    def __init__(self):
        pass

    def _get_tasks(self, **kwargs):
        pass

    def consume(self, url):
        pass


class SHHotelSpider(BaseSpider):
    # pre-defined
    identifier = 'hotel_spider'
    pd_columns = ["日期", "平均出租率", "五星级平均出租率", "平均房价", "五星级平均房价", "平均房价增长",
                  "五星级房价增长"]
    locs = ["星级饭店客房平均出租率", "五星级", "星级饭店平均房价", "五星级", "星级饭店平均房价增长", "五星级"]
    prefix = 'https://tjj.sh.gov.cn'
    start_url = 'https://tjj.sh.gov.cn/ydsj57/index.html'
    template = 'https://tjj.sh.gov.cn/ydsj57/index_{page}.html'

    def __init__(self, spider_status: Literal['new', 'update'], tasks: list = None):
        super().__init__()
        # need initialization
        self.tasks = tasks  # [[id,url],...]
        self.spider_status = spider_status  # new, update, resume, fetching tasks, stopped
        self.data = []
        self.task_pointer = 0  # 关键数据，spider排错的关键
        self.skipped_tasks = []

    def _get_tasks(self):
        def pattern_page(target: list):
            pattern = r'^共(\d+)页$'
            # 使用列表推导来筛选符合条件的元素
            # 使用 next() 函数和生成器表达式来找到第一个匹配的元素
            match = next((item for item in target if re.match(pattern, item)), None)
            if match is None:
                print("Warning, NO match, using default page=8")
                return 8
            else:
                k_value = int(match.group(1))  # 提取匹配的数字部分并转换为整数
                print("page=：", k_value)
                return k_value

        def pattern_urls(req_text):
            soup = BeautifulSoup(req_text, 'html.parser')
            # 查找第一个ul标签
            ul_tag = soup.find('ul')
            tmp_list = []
            if ul_tag:
                # 在这个ul标签内查找所有的li标签
                li_tags = ul_tag.find_all('li')
                for li in li_tags:
                    a_tag = li.find('a')
                    if a_tag and 'href' in a_tag.attrs:
                        tmp_list.append(self.prefix + a_tag['href'])
            return tmp_list

        if self.spider_status == 'resume':  # skip
            self.spider_status = 'running'
            return
        elif self.spider_status == 'new':
            self.spider_status = 'fetching tasks'
            # # request num k(8+)
            q0 = request_with_retry(self.start_url)
            k = pattern_page(get_pure_text(q0.text))
            for i in tqdm(range(1, k + 1)):
                if i == 1:
                    self.tasks.append(*pattern_urls(q0.text))
                else:
                    qi = request_with_retry(self.template.format(page=i))
                    self.tasks.append(*pattern_urls(qi.text))

            # 将UUID转换为字符串并去掉连字符，然后截取前8个字符
            ids = [self.identifier + '_' + str(uuid.uuid4()).replace('-', '')[:8] for _ in range(len(self.tasks))]
            self.tasks = zip(ids, self.tasks)
            self.spider_status = 'running'
            return self.tasks

        elif self.spider_status == 'update':
            self.spider_status = 'fetching tasks'
            # request_num 1
            q0 = request_with_retry(self.start_url)
            new_tasks = pattern_urls(q0.text)
            # 将列表转换为集合
            set1 = set(new_tasks)
            set2 = set(self.tasks)
            # 使用差集找出list1中有而list2中没有的元素
            difference = list(set1.difference(set2))
            ids = [self.identifier + '_' + str(uuid.uuid4()).replace('-', '')[:8] for _ in range(len(difference))]
            self.tasks = list(zip(ids, self.tasks))
            self.spider_status = 'running'
            return self.tasks

    def consume(self, url):
        def pattern_data(loc_str: str, start_idx: int, p_text_list: list[str]):
            # 找到两个index后面的第一个数字
            # 如果找到了相应的索引，查找该索引后的第一个看起来像是浮点数的字符串
            index_s1 = next(
                (i for i in range(start_idx, len(p_text_list)) if loc_str in p_text_list[i]))
            index_n1 = next(
                (i for i in range(index_s1, len(p_text_list)) if
                 re.match(r'^-?\d+(\.\d+)?$', p_text_list[i])),
                None)
            if index_n1 is None:
                print("在指定索引之后没有找到浮点数")
                return None, None
                # 将找到的数字字符串转换为浮点数
            number = float(p_text_list[index_n1])
            return number, index_n1

        # datetime pattern
        def pattern_date(p_text_list: list[str]):
            # 正则表达式匹配 xxxx年x月
            pattern = r'(\d{4})年(\d{1,2})月'
            # 使用 next() 和生成器表达式找到第一个匹配的字符串，并提取年和月
            year, month = next(
                (re.search(pattern, item).groups() for item in p_text_list if re.search(pattern, item)), (None, None))
            if year and month:
                return date(day=1, month=int(month), year=int(year))
            else:
                print("Warning: No year or month")
                return None

        starter = 0
        data = []
        query = request_with_retry(url)
        txt = get_pure_text(query.text)
        data.append(pattern_date(txt))
        for i in self.locs:
            # 每次更新起点index
            n1, starter = pattern_data(i, starter, txt)
            data.append(n1)

        assert len(data) == len(self.locs) + 1  # 1 for date

        return data

    def run(self):
        if self.spider_status == 'stopped':
            print('Stopped spider cannot use run, please use spider.resume() instead.')
            return
        try:
            self._get_tasks()
            try:
                for i in range(len(self.tasks)):
                    # task[] = [ids, url]
                    data = self.consume(self.tasks[i][1])
                    # 上一步不出错，下两步必然执行完成，出错了，则立即转为pkl准备排错恢复
                    self.data.append(data)
                    self.task_pointer = i + 1  # 表示正在执行该任务
            except Exception as e:
                self.dump()
                print(
                    f'Error occurred while processing tasks, last task is {self.data[-1]}, spider dumped with error: {e}')
        except Exception as e:
            print('Error in get tasks:', e)

    def dump(self):
        self.spider_status = 'stopped'
        with open(f"{self.identifier}_dumped_{str(uuid.uuid4()).replace('-', '')[:8]}.pkl", 'wb') as file:
            pickle.dump(self, file)

    def resume(self):
        self.spider_status = 'resume'
        self.run()

    def skip(self, num=1):
        for i in range(num):
            # taskpointer表示正在处理，skip掉则直接skip对应掉task，然后加一taskpointer
            self.skipped_tasks.append(self.tasks[self.task_pointer])
            self.task_pointer += 1

    def data(self):
        return pd.DataFrame(self.data, columns=self.pd_columns)


class HKVisitorsSpider(BaseSpider):
    def __init__(self):
        super().__init__()
        self.tasks = []
        self.data = None
        self.db_table = ''

    def _get_tasks(self):
        self.tasks = 'https://www.immd.gov.hk/opendata/hkt/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv'

    def consume(self, url):
        def download_and_load_csv(link):
            headers = {
                'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7'  # 优先请求简体中文，如果不可用则请求其他中文变体
            }
            # 使用requests下载CSV文件内容
            response = request_with_retry(link, custom_headers=headers)
            # 将下载的内容解码为字符串，然后使用pandas读取
            from io import StringIO
            data = StringIO(response.text)
            return pd.read_csv(data)

        df = download_and_load_csv(url)
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

        # 打印结果
        return grouped

    def run(self):
        # only 1 task
        self._get_tasks()
        self.data = self.consume(self.tasks)

    def data(self):
        return self.data


# unit test
if __name__ == '__main__':
    database_url = "sqlite:///./data.sqlite"
    engine = create_engine(database_url)
    query = f"select url from spd_tasks where unique_id like '%{SHHotelSpider.identifier}%'"
    df = pd.read_sql_query(query, engine)
    tasks = df['url'].tolist()

    spider = SHHotelSpider('update', tasks)
    # spider.run()
    spider.consume('https://tjj.sh.gov.cn/ydsj57/20231116/d474b2c8bb5647f2a4041299caad8be7.html')

    spider = HKVisitorsSpider()
    spider.run()
    print(spider.data)
