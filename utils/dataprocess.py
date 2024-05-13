import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sqlalchemy import create_engine


def plot_time_series(df, title='Time Series Data', xlabel='Date', ylabel='Value', figsize=(10, 6)):
    """
    绘制一个或多个时序数据的函数。

    参数:
    - df: 一个pandas DataFrame，索引为日期时间类型，每一列是一个时间序列。
    - title: 图表的标题。
    - xlabel: x轴的标签。
    - ylabel: y轴的标签。
    - figsize: 图表的尺寸，以英寸为单位。
    """
    plt.figure(figsize=figsize)
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


# melt and pivot are all implemented by django_pandas

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

def pivot(df):
    return df


def impute(df):
    # data has already been imputed
    return df


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
    for year in range(datetime.date.fromisoformat(regular_slice_start).year,
                      datetime.date.fromisoformat(regular_slice_end).year + 1):
        # 复制2019年数据，更新年份
        df_temp = hk_regular_slice.copy()
        df_temp.index = df_temp.index.map(lambda x: x.replace(year=year))

        # 处理闰年，添加2月29日
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):  # 判断闰年
            feb_28 = df_temp.loc[f'{year}-02-28']
            # 插入2月29日，比例与前一天相同
            df_temp = pd.concat([
                df_temp[:f'{year}-02-28'],
                pd.DataFrame({'global_airport_entry': [feb_28['global_airport_entry']]},
                             index=[pd.Timestamp(f'{year}-02-29')]), df_temp[f'{year}-02-29':]
            ])

        frames.append(df_temp)

    # 合并所有年份的DataFrame
    hk_former = pd.concat(frames)
    hk_former += np.random.normal(0, np.std(hk_former.values) * noise, hk_former.shape)
    hk_all = pd.concat([hk_former, hk_special_slice])

    # 计算hk ratio
    reg_ratio = pd.DataFrame(index=hk_all.index)
    for column in hk_all.columns:
        # 对于每个月，计算该月每一天的值占该月总和的比例
        reg_ratio[column] = hk_all[column] / hk_all[column].resample('ME').transform('sum')

    sh_daily_interpolated = sh_daily_fake.multiply(reg_ratio['global_airport_entry'].head(len(sh_daily_fake)),
                                                   axis=0).astype('int')
    return sh_daily_interpolated


def cut(df, start_date='2020-01-01', end_date='2023-06-01'):
    # 使用布尔索引选择不在指定范围内的数据
    df = df.loc[(df.index < start_date) | (df.index > end_date)].sort_index()
    # 创建一个递减的日期索引
    new_index = pd.date_range(end=df.index.max(), periods=len(df))
    # 将新的日期索引应用到数据框架
    df.set_index(new_index, inplace=True)
    # 使用布尔索引选择不在指定范围内的数据
    return df


# unit test
if __name__ == '__main__':
    database_url = "sqlite:///D:/lib/Travel_ML/data/data.sqlite"
    engine = create_engine(database_url)

    imputed_data_query = f"SELECT date,global_airport_entry FROM hk_visitors_imputed"
    df_hk = pd.read_sql_query(imputed_data_query, engine, index_col='date', parse_dates=['date'])
    imputed_data_query = f"SELECT * FROM sh_visitors"
    df_sh = pd.read_sql_query(imputed_data_query, engine, index_col='date', parse_dates=['date'])
    sh_daily = interpolation(df_sh, df_hk, noise=0.1)
    p = melt(sh_daily)
    print(p)

    data = spider
