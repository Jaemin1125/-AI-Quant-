import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#데이터 전처리
data=pd.read_csv('C://Users//user//Documents//drive-download-20260427T041452Z-3-001/OBS_ASOS_MI_20241216173201(1).csv', encoding='cp949')
print(data)

data.rename(columns={'기온(°C)': 'Temperature', '일시':'Date'}, inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

data['Temperature'] = pd.to_numeric(data['Temperature'], errors='coerce')

print('데이터 전처리 완료:')

print(data.head())


print(f"초기 결측값 수: {data['Temperature'].isnull().sum()}개")

#물리 한계 검사
min_temp_limit = -40
max_temp_limit = 50

data.loc[(data['Temperature']<min_temp_limit) | (data['Temperature']>max_temp_limit), 'Temperature'] = np.nan

print(f"물리 한계 검사 후 결측값 수: '{data['Temperature'].isnull().sum()}개")

#단계 검사
step_threshold = 5

data['temp_diff'] = data['Temperature'].diff().abs()
data.loc[data['temp_diff'] > step_threshold, 'Temperature'] = np.nan
data.drop(columns=['temp_diff'], inplace=True)

print(f"단계 검사 후 결측값 수: {data['Temperature'].isnull().sum()}개")

#지속성 검사
persistence_threshold = 0.1
window_size_minutes = 60

abs_diff = data['Temperature'].diff().abs().fillna(0)

rolling_sum_abs_diff = abs_diff.rolling(window = f"{window_size_minutes}min", min_periods=window_size_minutes).sum()

is_persistent = (rolling_sum_abs_diff < persistence_threshold)

invalidation_mask = pd.Series(False, index=data.index)
for i in data.index[is_persistent]:
    start_of_window = i - pd.Timedelta(minutes = window_size_minutes - 1)
    invalidation_mask.loc[start_of_window:i] = True

data.loc[invalidation_mask, 'Temperature'] = np.nan

print(f"지속성 검사 후 결측값 수 : {data['Temperature'].isnull().sum()}개")

#자료 완전성 검사 및 시간별/일별 평균 산출
completeness_threshold_ratio = 0.8  #80%자료 완전성 임계값

hourly_data = data['Temperature'].resample('h')
hourly_count = hourly_data.count()
hourly_expected_count = 60
hourly_mean = hourly_data.mean()
hourly_mean[hourly_count < hourly_expected_count * completeness_threshold_ratio]= np.nan

print('\n 시간별 평균(일부):')
print(hourly_mean.head())

#3시간 평균
three_hourly_data = data['Temperature'].resample('3h')
three_hourly_count = three_hourly_data.count()
three_hourly_expected_count = 3*60
three_hourly_mean = three_hourly_data.mean()
three_hourly_mean[three_hourly_count < three_hourly_expected_count * completeness_threshold_ratio] = np.nan

print('\n 시간별 평균(일부): ')
print(three_hourly_mean.head())

#일 평균
daily_data = data['Temperature'].resample('d')
daily_count = daily_data.count()
daily_expected_count = 24*60
daily_mean = daily_data.mean()
daily_mean[daily_count < daily_expected_count * completeness_threshold_ratio] = np.nan

print('\n 일별 평균(일부): ')
print(daily_mean.head())
