import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc 
import dash_html_components as html
from scipy.stats import linregress
from dateutil.relativedelta import relativedelta

from pandas_datareader import data as web 
from datetime import datetime as dt

def conver_run_to_time_string(run:int):
    hours = run // 60
    minutes = run % 60
    return f'{hours:02}:{minutes:02}:00'


def make_variational_dataset(
	date_start:str='01-01-2018', 
	date_end:str='01-01-2021', 
	format_str:str='%d-%m-%Y',
	prefix:str='',
	det_num:int=4,
	sensors:list=[],
	usecols:list=[0, 1],
	path_prefix:str='',
	save_to:str=''):
	"""
	function to combine variational data produced by en-detector arrays to pandas dataset
	
	Parameters:
	- date_start : date from that to make output dataset, default = '01-01-2018'
	- data_end : date up to taht to make output dataset, default = '01-01-2021'
	- format_str : format string for date_start and date_end, default = '%d-%m-%Y'
	- prefix : prefix of the variational data files, default = '' (empty string)
	- det_num : number of detectors in files, default = 4
	- sensors : list of sensor column names, default = [] (empty list)
	- usecols : list of columns to use, default = [0, 1]
	- path_prefix : path to the folder with variational data files
	- save_to : output file path
	
	return value : pandas dataset
	"""
							
	date_start, date_end = pd.to_datetime([date_start, date_end], format=format_str)
	date_range = [date_start + pd.Timedelta(days=i) for i in range((date_end - date_start).days + 1)]

	colnames = ['run'] + ['n' + str(i) for i in range(1, det_num + 1)] + \
				['ch' + str(i) for i in range(1, det_num + 1)] +\
					sensors
	  
	data = None
	start = 0
	for i in date_range:
		filename = f'{prefix}{i:%m-%d.%y}'
		if (os.path.isfile(path_prefix + filename)):
			tmp = pd.read_csv(path_prefix + filename, delim_whitespace = True, usecols=usecols, 
                              names = colnames)
			i1, i2, i3 = len(prefix), len(prefix) + 3, len(prefix) + 6
			tmp['time'] = tmp.run.apply(conver_run_to_time_string)
			tmp.time = pd.to_datetime(filename[i1:i1+2] + '.' + filename[i2:i2+2] + \
									  '.' + filename[i3:i3+2] + ' ' + tmp.time)
			tmp.index=tmp['time']
			if (start == 0):
				data = tmp
				start = 1
			else:
				data = data.append(tmp)
			del tmp
	data.drop(['time', 'run'], axis=1, inplace=True)
	if save_to != '':
		data.to_csv(save_to, sep='\t')
	return data


def make_eq_data_emsc_from_csv(filename:str, 
						  ncols:int=11,
						  drop:list=[]):
	"""
	function to make earthquake dataset from csv downloaded from EMSC
	
	Parameters:
	- filename : path to the csv file
	- ncols : number of columns to use from initial file, default = 11
	- drop : list of columns to drop (do not include here Date and Time UTC), default = []
	return value : pandas dataset
	"""
	eq = pd.read_csv(filename, delimiter=';', usecols=list(range(11)))
	eq['date_time'] = pd.to_datetime(eq['Date'] + ' ' + eq['Time UTC'], format='%Y-%m-%d')
	eq = eq.sort_values(by='date_time')
	eq.drop(['Date', 'Time UTC']+drop, axis=1,inplace=True)
	return eq


def make_eq_data_kam_from_csv(filename:str, 
						  ncols:int=11,
						  drop:list=['Time UTC - 12h']):
	"""
	function to make earthquake dataset from csv downloaded from Kamchatcka GEO Service
	
	Parameters:
	- filename : path to the csv file
	- ncols : number of columns to use from initial file, default = 11
	- drop : list of columns to drop (do not include here Date and T_UTC), default = ['Time UTC - 12h']
	return value : pandas dataset
	"""
	eq = pd.read_csv(filename, delimiter=';', usecols=list(range(11)))
	eq['date_time'] = pd.to_datetime(eq['Date'][:10] + ' ' + eq['T_UTC'][:7], format='%d.%m.%Y %H:%M:%S')
	eq = eq.sort_values(by='date_time')
	eq.drop(['Date', 'T_UTC']+drop, axis=1,inplace=True)
	return eq


def clean_data_from_outliers(data:pd.DataFrame,
							 cols:list=[],
							 upper_bound_sigma:float=5,
							 lower_bound_sigma:float=5):
	"""
	function to clean data from outliers
	
	Parameters:
	- data : pandas dataframe with data to clean
	- cols : list of column numbers to clean from outliers
	- upper_bound_sigma : upper bound to clean in terms of standard devations. All rows with values of considering columns above their mean + std * upper_bound_sigma will be deleted, default = 5
	- lower_bound_sigma : the same for lower bound, default = 5
	return value : new dataset cleaned from outliers
	"""
	print(f'initial size: {data.shape[0]}')
	if cols == []:
		cols = data.columns
	new_data = data.copy()
	for column in cols:
		size_before = new_data.shape[0]
		mean, sigma = new_data[column].mean(), new_data[column].std()
		new_data = new_data[new_data[column] > mean - lower_bound_sigma * sigma]
		new_data = new_data[new_data[column] < mean + upper_bound_sigma * sigma]
		size_after = new_data.shape[0]
		print(f'dropped for column ({column}) size: {size_before - size_after}')
	print(f'final size: {size_after}')
	return new_data


def resample_data(data:pd.DataFrame,
				  period:str='1d',
				  way:str='mean'):
	"""
	function to resample data with selected step size
	
	Parameters:
	- data : pandas dataframe with data to resample
	- period : step size, for example '1d', '3h', '30min' etc, default = '1d' (1 day)
	- way : way to resample, possible values are: 'mean', 'sum', 'std', default = mean
	return value : new resampled dataset
	"""
	if way == 'mean':
		return data.resample(period).mean().dropna()
	elif way == 'sum':
		return data.resample(period).sum().dropna()
	elif way == 'std':
		return data.resample(period).std().dropna()
	else:
		return data


def plot_time_series_plotly(data:pd.DataFrame,
					 cols:list=[],
					 date_start:str='',
					 date_end:str='',
					 format_str:str='%d-%m-%Y',
					 title:str="Time series",
					 xlabel:str="Date & Time",
					 ylabel:str="Counting rate"):
	"""
	function to plot time series data
	
	Parameters:
	- data : pandas dataframe with data to plot
	- cols : list of column numbers to plot, default = [] (empty list)
	- date_start : date to start plot, default = '' (empty string) means start from first date in the dataset
	- date_end : right limit date for the plot, default = ''
	- format_str : format string for date_start and date_end, default = '%d-%m-%Y'
	- title : title of the plot, default = 'Time series'
	- xlabel : label for the x axis, default = 'Date & Time'
	- ylabel : label for the y axis, default = 'Counting rate'
	no return value
	"""
	date_start, date_end = pd.to_datetime([date_start, date_end], format=format_str)
	df = data[data.index > date_start]
	df = df[df.index < date_end]
	if cols == []:
		cols = df.columns
	fig = go.Figure()
	for col in cols:
		fig.add_trace(go.Scatter(x=df.index, y=df[col],
                    mode='lines',
                    name=col))
	fig.update_layout(
		title=title,
		xaxis_title=xlabel,
		yaxis_title=ylabel,
		legend_title="Legend",
		font=dict(
			family="Courier New, monospace",
			size=18,
			color="RebeccaPurple"
		)
	)
	fig.show()
	
	
def plot_correlation_plotly(data:pd.DataFrame,
					 x_parameter:str='',
					 normalize:bool=True,
					 fit:bool=True,
					 cols:list=[],
					 date_start:str='',
					 date_end:str='',
					 format_str:str='%d-%m-%Y',
					 title:str="Correlation",
					 xlabel:str="",
					 ylabel:str=""):
	"""
	function to plot correlations
	
	Parameters:
	- data : pandas dataframe with data to plot
	- x_parameter : column name to be x axis of the correlation plot, default = '' (empty string)
	- normalize : to normalize y axis data or not, True or False, default = 'True'
	- fit : to fit correlation dependence or not, True or False, default = 'True'
	- cols : list of column numbers to plot, default = [] (empty list)
	- date_start : date to start plot, default = '' (empty string) means start from first date in the dataset
	- date_end : right limit date for the plot, default = ''
	- format_str : format string for date_start and date_end, default = '%d-%m-%Y'
	- title : title of the plot, default = 'Time series'
	- xlabel : label for the x axis, default = 'Date & Time'
	- ylabel : label for the y axis, default = 'Counting rate'
	no return value
	"""
	if x_parameter == '' or x_parameter not in data.columns:
		print('No x parameter specified or it is invalid')
		return
	date_start, date_end = pd.to_datetime([date_start, date_end], format=format_str)
	df = data[data.index > date_start]
	df = df[df.index < date_end]
	if df.shape[0] > 1e4:
		df = df.resample('1h').mean()
	if cols == []:
		cols = df.columns
	fig = go.Figure()
	for col in cols:
		y = df[col]
		if normalize:
			y /= df[col].mean()
		fig.add_trace(go.Scatter(x=df[x_parameter], y=y,
                    mode='markers',
                    name=col))
		if fit:
			k, b, r_value, p_value, std_err = linregress(df[x_parameter], y)
			x = np.linspace(df[x_parameter].min(), df[x_parameter].max(), 100)
			fig.add_trace(go.Scatter(x=x, y=k*x+b,
                    mode='lines',
                    name=col + ' trendline'))
			print(f'{col}: {k * 100: .3f}(%/mm Hg) * x + {b:.2f}, r-value = {r_value:.3f}')
	fig.update_layout(
		title=title,
		xaxis_title=xlabel,
		yaxis_title=ylabel,
		legend_title="Legend",
		font=dict(
			family="Courier New, monospace",
			size=18,
			color="RebeccaPurple"
		)
	)
	fig.show()
		
		

	
def correct_data_for_p(data:pd.DataFrame,
					   date_start:str='',
					   date_end:str='',
					   format_str:str='%d-%m-%Y',
					   cols:list=[],
					   p_name:str='P',
					   period_in_months:int=3):
	"""
	function to correct data for pressure
	
	Parameters:
	- data : pandas dataframe with data to correct, must have pressure data
	- date_start : start date for output dataset, default = '', means start point of the data
	- date_end : end date for output dataset, default = '', means final point of the data
	- format_str : format string for the date_start/date_end, default = '%d-%m-%Y'
	- cols : list of columns to be corrected, default = [] (empty list)
	- p_name : name of the pressure column, default = 'P'
	- period_in_month : number of months to determine barometric coefficient and to correct the data for it
	return value : new dataset with selected columns corrected for pressure
	"""
	if date_start != '' and date_end != '':
		date_start, date_end = pd.to_datetime([date_start, date_end], format=format_str)
		if date_end > data.index[-1]:
			date_end = data.index[-1]
	else:
		date_start = data.index[0]
		date_end = data.index[-1]
	df = data[data.index >= date_start]
	df = df[df.index <= date_end]
	date_cur = date_start
	while (date_cur <  date_end):
		date_tmp = date_cur + relativedelta(months=period_in_months)
		df_tmp = df[df.index >= date_cur]
		df_tmp = df_tmp[df_tmp.index <= date_tmp]
		for col in cols:
			y = df_tmp[col]
			k, b, r_value, p_value, std_err = linregress(df_tmp[p_name], y / y.mean())
			df_tmp[col + ' corrected'] = df_tmp[col] * (1. + k * (df_tmp[p_name].mean() - df_tmp[p_name]))
		if (date_cur == date_start):
			out_data = df_tmp
		else:
			out_data = out_data.append(tmp_data)
		date_cur = date_tmp
	return out_data


def epoch_analysis(
        data: pd.DataFrame(),
        date_start:str='',
		date_end:str='',
		format_str:str='%d-%m-%Y',
        period: str='1d',
        step_size: str='1h'
        ):
	"""
	fuction to make superimposed epoch analysis
	
	Parameters:
	- data : pandas dataframe with data to make analysis
	- date_start : start date for output dataset, default = '', means start point of the data
	- date_end : end date for output dataset, default = '', means final point of the data 
	- format_str : format string for the date_start/date_end, default = '%d-%m-%Y'
	- period : length of the epoch, default = '1d'
	- step_size : step size of data, default = '1h'
	return value : new dataset with superimposed epochs
	"""
	dt = data.resample(step_size).mean()
	if date_start != '' and date_end != '':
		date_start, date_end = pd.to_datetime([date_start, date_end], format=format_str)
	else:
		date_start = data.index[0]
		date_end = data.index[-1]
	date_rng = pd.date_range(start=date_start, end=date_end, freq=period)
	df_out = dt[date_rng[0] : date_rng[1]][:-1]
	df_out.index = np.linspace(0, df_out.shape[0] - 1, df_out.shape[0])
	for i in range(len(date_rng[1:-1])):
		df_tmp = dt[date_rng[i] : date_rng[i + 1]][:-1]
		df_tmp.index = np.linspace(0, df_tmp.shape[0] - 1, df_tmp.shape[0])
		df_out += df_tmp
	df_out = df_out / len(date_rng)
	return df_out


def plot_epochs_plotly(data:pd.DataFrame,
		       cols:list=[],
		       figsize:tuple=(),
		       normalize:bool=True,
		       title:str="Superimposed epoch analysis",
		       xlabel:str="Time",
		       ylabel:str="Counting rate"):
	"""
	function to plot superimposed epoch method results
	
	Parameters:
	- data : pandas dataframe with data to plot
	- cols : list of column numbers to plot, default = [] (empty list)
	- normalize : to normalize y axis data or not, True or False, default = 'True'
	- title : title of the plot, default = 'Time series'
	- xlabel : label for the x axis, default = 'Date & Time'
	- ylabel : label for the y axis, default = 'Counting rate'
	no return value
	"""
	df = data
	if cols == []:
		cols = df.columns
	fig = go.Figure()
	for col in cols:
		if normalize:
			delim = df[col].mean()
		else:
			delim = 1
		fig.add_trace(go.Scatter(x=df.index, y=df[col] / delim,
                    mode='lines',
                    name=col))
	fig.update_layout(
		title=title,
		xaxis_title=xlabel,
		yaxis_title=ylabel,
		legend_title="Legend",
		font=dict(
			family="Courier New, monospace",
			size=18,
			color="RebeccaPurple"
		)
	)
	fig.show()


def make_table_of_eq_data_for_epoch(
        dataset: pd.DataFrame,
        days_before: int = 5,
        days_after:  int = 5,
        events: list = [],
        column: str = '',
        format_str: str = '%d-%m-%Y %H:%M:%S',
        time_step: str = '3h',
        normalize = True
        ):
    """
    Function to make dataset where each column is rate around one event

    dataset - data with en-detector rate
    days_before - number of days before
    days_after - number of days after
    events - list of event datetimes
    column - name of column in initial dataset
    format_str - format of event datetime
    """
    tmp_data = dataset.resample(time_step).mean()
    out_data = []
    if type(events[0]) == 'str':
        events = pd.to_datetime(events, format=format_str)
    name_dict = {}
    i = 0
    for date in events:
        delta_before = pd.Timedelta(str(days_before) + ' days')
        delta_after = pd.Timedelta(str(days_after) + ' days')
        left_border = date - delta_before
        right_border = date + delta_after
        if left_border < tmp_data.index.min() or \
           right_border > tmp_data.index.max():
            continue
        tmp = tmp_data[tmp_data.index > left_border]
        tmp = tmp[tmp.index < right_border][column]
        new_index = list(range(
                                -1 * int(tmp.shape[0] * (days_before / (days_before + days_after))),
                                int(tmp.shape[0] * (days_after / (days_before + days_after)))
                            ))
        tmp.index = new_index
        if normalize:
            tmp /= tmp.mean()
        out_data.append(tmp)
        name_dict[i] = str(date)[:16]
        i += 1
    out_data = pd.concat(out_data, ignore_index=True, axis=1)
    out_data.rename(name_dict, axis=1, inplace=True)
    return out_data
