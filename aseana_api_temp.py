#!/usr/bin/env python
# encoding: utf-8

import json
from http import HTTPStatus
from flask import Flask, request, jsonify, abort

# libraries used for the prediction
import pandas as pd
import numpy as np
import plotly.express as px
from pycaret.regression import create_model, predict_model, finalize_model, setup, compare_models

# custom
from constants import *

app = Flask(__name__)


@app.route('/')
def index():
    return 'Aseana Project API'


@app.route('/api/v1/predict', methods=['POST'])
def predict():

    # check if the headers content-type is json format
    if request.content_type != 'application/json':
        return jsonify({'status': HTTPStatus.BAD_REQUEST, 'message': HTTPStatus.BAD_REQUEST.description})

    # get the request payload
    request_json_data = request.get_json()

    # check if the request has a json payload
    if request_json_data is None or len(request_json_data) != len(JSON_DATA):
        return jsonify({'status': HTTPStatus.BAD_REQUEST, 'message': HTTPStatus.BAD_REQUEST.description})

    # created auth token here since instead of the headers since we don't use and expect super high security measures
    auth_token = request_json_data[JSON_DATA[0]]

    if auth_token != AUTH_TOKEN:
        return jsonify({'status': HTTPStatus.UNAUTHORIZED, 'message': HTTPStatus.UNAUTHORIZED.description})
    column_name = request_json_data[JSON_DATA[1]]
    years_to_predict = request_json_data[JSON_DATA[2]]
    data_to_predict = request_json_data[JSON_DATA[3]]
    # translate the json data to pandas dataframe and catch some error on translating
    try:
        data_to_predict_df = pd.DataFrame.from_records(data_to_predict)
    except Exception as e:
        return jsonify({'status': HTTPStatus.INTERNAL_SERVER_ERROR, 'message': e})
    # catch errors on prediction
    try:
        # call the prediction method
        prediction_data = create_prediction(data_to_predict_df, years_to_predict, column_name)
    except Exception as e:
        return jsonify({'status': HTTPStatus.INTERNAL_SERVER_ERROR, 'message': e})

    return jsonify({'status': HTTPStatus.OK, 'prediction_data': prediction_data})


def create_prediction(data, years_to_predict, column_name):

    # convert the date string into datetime
    data['Date'] = pd.to_datetime(data['Date'])
    
    # extract month and year from dates
    data['Month'] = [i.month for i in data['Date']]
    data['Year'] = [i.year for i in data['Date']]
    # create a sequence of numbers
    data['Series'] = np.arange(1, len(data) + 1)
    # drop unnecessary columns and re-arrange
    data.drop(['Date'], axis=1, inplace=True)
    data = data[['Series', 'Year', 'Month', column_name]]
    start_year = list(data.iloc[[0, ]]['Year'])[0]
    end_year = list(data.iloc[[-1, ]]['Year'])[0]
    # split data into train-test set
    yd = end_year - start_year
    # get the test percentage 20%
    tep = int(yd * 0.2)
    # get the train percentage 80%
    trp = yd - tep
    train = data[data['Year'] < (end_year - tep)]
    test = data[data['Year'] >= (end_year - tep)]
    # initialize configuration
    prediction_config = setup(data=train,
                              test_data=test,
                              target=column_name,
                              fold_strategy='timeseries',
                              numeric_features=['Year', 'Series'],
                              fold=3,
                              use_gpu=True,
                              transform_target=True,
                              session_id=123,
                              html=False,
                              silent=True,
                              verbose=False)
    # select fitting model / preprocessing
    best = compare_models(sort='MAE', verbose=False)
    # generate predictions on the original dataset
    final_best = finalize_model(best)
    psy = '{}-01-01'.format(end_year + 1)
    pey = '{}-01-01'.format(end_year + years_to_predict)
    future_dates = pd.date_range(start=psy, end=pey, freq='MS')

    # create a pandas dataframe holder for the prediction
    future_df = pd.DataFrame()
    future_df['Month'] = [i.month for i in future_dates]
    future_df['Year'] = [i.year for i in future_dates]
    future_df['Series'] = np.arange(len(data.index), (len(data.index) + len(future_dates)))

    # create the prediction that return a pandas dataframe
    predictions_future = predict_model(final_best, data=future_df)

    # change the column names
    predictions_future.columns = ['Month', 'Year', 'Series', '{} Prediction'.format(column_name)]

    predictions_future_to_json = predictions_future.copy()

    # merge the date (year-month)
    predictions_future_to_json['Date'] = pd.to_datetime(predictions_future_to_json[['Year', 'Month']].assign(DAY=1)).dt.strftime('%Y-%m')

    # remove some column
    predictions_future_to_json.drop(['Series'], axis=1, inplace=True)
    predictions_future_to_json.drop(['Year'], axis=1, inplace=True)
    predictions_future_to_json.drop(['Month'], axis=1, inplace=True)

    # convert predictions to int
    predictions_future_to_json = predictions_future_to_json.astype({'{} Prediction'.format(column_name): int})

    # translate the prediction to json format
    prediction_json = predictions_future_to_json.to_dict(orient='records')

    # debug for showing the graph result for the prediction
    if SHOW_GRAPH:
        psy = '{}-01-01'.format(start_year)
        concat_df = pd.concat([data, predictions_future], axis=0)
        concat_df_i = pd.date_range(start=psy, end=pey, freq='MS')
        concat_df.set_index(concat_df_i, inplace=True)
        fig = px.line(concat_df, x=concat_df.index, y=[column_name, '{} Prediction'.format(column_name)], template='plotly_dark', markers=True)
        fig.show()

    return prediction_json


# app.run(debug=True)
