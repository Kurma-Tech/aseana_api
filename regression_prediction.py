import pandas as pd
import numpy as np

data = pd.read_csv('AirPassengers.csv')
data['Date'] = pd.to_datetime(data['Date'])

print(data.head())

# create 12 month moving average
data['MA12'] = data['Passengers'].rolling(12).mean()

# plot the data and MA
import plotly.express as px
# fig = px.line(data, x="Date", y=["Passengers", "MA12"], template = 'plotly_dark')
# fig.show()

# extract month and year from dates
data['Month'] = [i.month for i in data['Date']]
data['Year'] = [i.year for i in data['Date']]

# create a sequence of numbers
data['Series'] = np.arange(1,len(data)+1)

# drop unnecessary columns and re-arrange
data.drop(['Date', 'MA12'], axis=1, inplace=True)
data = data[['Series', 'Year', 'Month', 'Passengers']] 

# split data into train-test set
train = data[data['Year'] < 1960]
test = data[data['Year'] >= 1960]

# import the regression module
from pycaret.regression import *
print("here")
# initialize setup
s = setup(data = train, test_data = test, target = 'Passengers', fold_strategy = 'timeseries', numeric_features = ['Year', 'Series'], fold = 3, transform_target = True, html=False, silent=True, session_id = 123, verbose=False)
print("here")
best = compare_models(sort = 'MAE', verbose=False)

# prediction_holdout = predict_model(best)
print("here")
# generate predictions on the original dataset
predictions = predict_model(best, data=data)
# add a date column in the dataset
predictions['Date'] = pd.date_range(start='1949-01-01', end = '1960-12-01', freq = 'MS')
print("here")
print(predictions['Date'])
# line plot
fig = px.line(predictions, x='Date', y=["Passengers", "Label"], template = 'plotly_dark')
# add a vertical rectange for test-set separation
fig.add_vrect(x0="1960-01-01", x1="1960-12-01", fillcolor="grey", opacity=0.25, line_width=0)
fig.show()