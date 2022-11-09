import mysql.connector
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from pycaret.regression import *
from http import HTTPStatus


app = Flask(__name__)

@app.route('/')
def index():
    return 'Aseana Project API'

@app.route('/api/v1/predict', methods=['POST'])
def predict():
  try:
    mydb = mysql.connector.connect(
      host="13.250.218.164",
      # host="localhost",
      port="3306",
      user="asean",
      # user="root",
      password="Zenith4780@",
      # password="",
      # database="asean"
      database="aseana_db"
    )

    mycursor = mydb.cursor()

    queryString = "SELECT month_and_year, COUNT(*) FROM businesses "
    request_json_data = request.get_json()

    if (request_json_data["country_id"] is not None):
      queryString = queryString + f"WHERE country_id = {request_json_data['country_id']} "

    if (request_json_data["classification_id"] is not None):
      queryString = queryString + f"AND parent_classification_id = {request_json_data['classification_id']} "

    queryString = queryString + "GROUP BY month_and_year ORDER BY month_and_year"

    mycursor.execute(queryString)

    myresult = mycursor.fetchall()

    idx = pd.date_range('2011-01-01', '2019-12-01', freq = 'MS')

    dataframe = pd.DataFrame()
    dataframe['DateTime'] = [i[0] for i in myresult]

    dataframe['Count'] = [i[1] for i in myresult]

    dataframe['DateTime'] = pd.to_datetime(dataframe['DateTime']).dt.date
    dataframe.sort_values(by="DateTime", inplace=True)

    dataframe.index = pd.DatetimeIndex(dataframe['DateTime'])

    dataframe = dataframe.reindex(idx, fill_value=0)

    # dataframe['DateTime'] = dataframe.index
    dataframe.drop(['DateTime'], axis=1, inplace=True)
    dataframe = dataframe.rename_axis('DateTime').reset_index()

    # print(dataframe)


    # create 12 month moving average
    dataframe['MA12'] = dataframe['Count'].rolling(12).mean()

    # plot the data and MA
    
    # fig = px.line(data, x="Date", y=["Passengers", "MA12"], template = 'plotly_dark')
    # fig.show()

    # extract month and year from dates
    dataframe['Month'] = [i.month for i in dataframe['DateTime']]
    dataframe['Year'] = [i.year for i in dataframe['DateTime']]

    # create a sequence of numbers
    dataframe['Series'] = np.arange(1,len(dataframe)+1)

    # drop unnecessary columns and re-arrange
    dataframe.drop(['DateTime', 'MA12'], axis=1, inplace=True)
    dataframe = dataframe[['Series', 'Year', 'Month', 'Count']] 

    # split data into train-test set
    train = dataframe[dataframe['Year'] < 2017]
    test = dataframe[dataframe['Year'] >= 2017]

    # import the regression module

    # initialize setup
    s = setup(data = train, test_data = test, target = 'Count', fold_strategy = 'timeseries', numeric_features = ['Year', 'Series'], fold = 3, transform_target = True, html=False, silent=True, session_id = 123, verbose=False)
    best = compare_models(sort = 'MAE', verbose=False)
    final_best = finalize_model(best)

    future_dates = pd.date_range(start = '2020-01-01', end = '2030-12-01', freq = 'MS')
    future_df = pd.DataFrame()
    future_df['Month'] = [i.month for i in future_dates]
    future_df['Year'] = [i.year for i in future_dates]    
    future_df['Series'] = np.arange(145,(145+len(future_dates)))
    future_df.head()

    predictions_future = predict_model(final_best, data=future_df)
    predictions_future.head()

    concat_df = pd.concat([dataframe,predictions_future], axis=0)
    concat_df_i = pd.date_range(start='2011-01-01', end = '2030-12-01', freq = 'MS')
    # concat_df["DateTime"] = concat_df_i
    # concat_df.set_index(concat_df_i, inplace=True)

    concat_df.drop(['Series'], axis=1, inplace=True)
    concat_df.drop(['Year'], axis=1, inplace=True)
    concat_df.drop(['Month'], axis=1, inplace=True)

    concat_df["Count"] = concat_df["Count"].fillna(0)
    concat_df["Label"] = concat_df["Label"].fillna(0)

    concat_df["Count"] = concat_df["Count"].astype(int)
    concat_df["Label"] = concat_df["Label"].astype(int)
    # print(concat_df)

    concat_df["Count"] = concat_df["Count"]+concat_df["Label"]
    # concat_df.drop(['Label'], axis=1, inplace=True)

    # convert predictions to int
    # concat_df = concat_df.astype({'Count': int})

    # translate the prediction to json format
    # prediction_json = concat_df["Count"].to_list()
    # print(concat_df["Count"])

    return jsonify({'status': HTTPStatus.OK, 'prediction_data': {
      "values": concat_df["Count"].to_list(),
      "keys":  concat_df_i.strftime('%Y-%m-%d').to_list()
    }})

    # fig = px.line(concat_df, x=concat_df.index, y=["Count", "Label"], template = 'plotly_dark')
    # fig.show()
  except:
    return jsonify({'status': HTTPStatus.INTERNAL_SERVER_ERROR, 'message': "Not enough data to forcast."})

app.run(debug=True)