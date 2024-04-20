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
from collections import deque

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36',
    # 更多用户代理字符串...
]


def request_with_retry(url, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        # 随机选择一个用户代理
        user_agent = random.choice(user_agents)
        headers = {'User-Agent': user_agent}

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

    def get_tasks(self, **kwargs):
        pass

    def consume(self, url):
        pass


class HotelSpider(BaseSpider):
    def __init__(self, spider_type: Literal['new', 'update'], tasks: list):
        super().__init__()
        self.identifier = 'hotel'
        self.tasks = tasks
        self.spider_type = spider_type
        self.data = pd.DataFrame(
            columns=["日期", "客房平均出租率", "五星级平均出租率", "平均房价", "五星级平均房价", "平均房价增长",
                     "五星级房价增长"])
        self.locs = ["星级饭店客房平均出租率", "五星级", "星级饭店平均房价", "五星级", "星级饭店平均房价增长", "五星级"]
        self.prefix = 'https://tjj.sh.gov.cn'
        self.start_url = 'https://tjj.sh.gov.cn/ydsj57/index.html'
        self.template = 'https://tjj.sh.gov.cn/ydsj57/index_{page}.html'

    def get_tasks(self):
        def pattern1(target: list):
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

        def pattern2(req_text):
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

        if self.spider_type == 'new':
            # request_num 1
            q0 = request_with_retry(self.start_url)
            k = pattern1(get_pure_text(q0.text))

            # request num k(8+)
            for i in tqdm(range(1, k + 1)):
                if i == 1:
                    pattern2(q0.text)
                else:
                    qi = request_with_retry(self.template.format(page=i))
                    pattern2(qi.text)

            # 生成一个随机UUID
            full_uuid = uuid.uuid4()
            # 将UUID转换为字符串并去掉连字符，然后截取前8个字符
            short_uuid = str(full_uuid).replace('-', '')[:8]
            ids = [self.identifier + short_uuid for _ in range(len(self.tasks))]
            self.tasks = pd.DataFrame([self.tasks, ids], columns=['id', 'url'])

            # df.to_sql('spider_links', con=self.dbEngine, if_exists='replace', index=False)
        elif self.spider_type == 'update':
            # request_num 1
            q0 = request_with_retry(self.start_url)
            k = pattern1(get_pure_text(q0.text))
            pattern2(q0.text)
            # 将列表转换为集合
            set1 = set(self.tasks)
            set2 = set()

            # 使用差集找出list1中有而list2中没有的元素
            difference = set1.difference(set2)

    def consume(self, url):
        def pattern1(locator: str, index_0: int, p_text_list: list[str]):
            # 找到两个index后面的第一个数字
            # 如果找到了相应的索引，查找该索引后的第一个看起来像是浮点数的字符串
            index_s1 = next(
                (i for i in range(index_0, len(p_text_list)) if locator in p_text_list[i]))
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
        def pattern2(p_text_list: list[str]):
            # 正则表达式匹配 xxxx年x月
            pattern = r'(\d{4})年(\d{1,2})月'
            # 使用 next() 和生成器表达式找到第一个匹配的字符串，并提取年和月
            year, month = next(
                (re.search(pattern, item).groups() for item in p_text_list if re.search(pattern, item)), (None, None))
            if year and month:
                return date(day=1, month=int(month), year=int(year))
            else:
                print("Warning: No year or month")

        starter = 0
        tmp = []
        query = request_with_retry(url)
        txt = get_pure_text(query.text)
        tm = pattern2(txt)
        tmp.append(tm)
        for i in self.locs:
            # 每次更新起点index
            n1, starter = pattern1(i, starter, txt)
            tmp.append(n1)
        self.data = pd.concat([self.data, pd.DataFrame([tmp], columns=self.data.columns)], ignore_index=True)
        # 防止数据丢失，返回已提取数据做持久化

    def run(self):
        for url in tqdm(self.tasks):
            self.consume(url)
            print('')


# unit test
if __name__ == '__main__':
    database_url = "sqlite:///../data/data.sqlite"
    engine = create_engine(database_url)

    spider = HotelSpider(db=engine, has_former_task_list=False)
    spider.consume('https://tjj.sh.gov.cn/ydsj57/20231116/d474b2c8bb5647f2a4041299caad8be7.html')
