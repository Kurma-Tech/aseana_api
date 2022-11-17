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
  # try:
    mydb = mysql.connector.connect(
      # host="13.250.218.164",
      host="localhost",
      # port="3306",
      # user="asean",
      user="root",
      # password="Zenith4780@",
      password="",
      database="asean"
      # database="aseana_db"
    )

    mycursor = mydb.cursor()

    request_json_data = request.get_json()

    queryString = f"SELECT month_and_year, COUNT(*) FROM {request_json_data['type']} "    

    if (request_json_data["country_id"] is not None):
      queryString = queryString + f"WHERE country_id = {request_json_data['country_id']} "

      if (request_json_data["classification_id"] is not None):
        queryString = queryString + f"AND parent_classification_id = {request_json_data['classification_id']} "
    else:
      if (request_json_data["classification_id"] is not None):
        queryString = queryString + f"WHERE parent_classification_id = {request_json_data['classification_id']} "

    queryString = queryString + "GROUP BY month_and_year ORDER BY month_and_year"

    mycursor.execute(queryString)

    myresult = mycursor.fetchall()

    # print(myresult)

    # idx = pd.date_range('2011-01-01', '2019-12-01', freq = 'MS')

    dataframe = pd.DataFrame()
    dataframe['DateTime'] = [i[0] for i in myresult]
    
    dataframe['Count'] = [i[1] for i in myresult]

    dataframe['DateTime'] = pd.to_datetime(dataframe['DateTime']).dt.date
    
    dataframe = dataframe.groupby('DateTime', as_index=False).sum()
    dataframe.sort_values(by="DateTime", inplace=True)

    dataframe.index = pd.DatetimeIndex(dataframe['DateTime'])

    # dataframe = dataframe.reindex(idx, fill_value=0)

    # dataframe['DateTime'] = dataframe.index
    dataframe.drop(['DateTime'], axis=1, inplace=True)
    dataframe = dataframe.rename_axis('DateTime').reset_index()

    # print(dataframe)


    # # create 12 month moving average
    # dataframe['MA12'] = dataframe['Count'].rolling(12).mean()

    # # plot the data and MA
    
    # # fig = px.line(data, x="Date", y=["Passengers", "MA12"], template = 'plotly_dark')
    # # fig.show()

    # extract month and year from dates
    dataframe['Month'] = [i.month for i in dataframe['DateTime']]
    dataframe['Year'] = [i.year for i in dataframe['DateTime']]

    # create a sequence of numbers
    dataframe['Series'] = np.arange(1,len(dataframe)+1)

    # drop unnecessary columns and re-arrange
    # dataframe.drop(['DateTime', 'MA12'], axis=1, inplace=True)
    dataframe = dataframe[['Series', 'Year', 'Month', 'Count']] 

    start_year = list(dataframe.iloc[[0, ]]['Year'])[0]
    end_year = list(dataframe.iloc[[-1, ]]['Year'])[0]
    # split data into train-test set
    yd = end_year - start_year
    # get the test percentage 20%
    tep = int(yd * 0.2)
    # get the train percentage 80%
    trp = yd - tep
    train = dataframe[dataframe['Year'] < (end_year - tep)]
    test = dataframe[dataframe['Year'] >= (end_year - tep)]

    # print(test)

    # # # split data into train-test set
    # # train = dataframe[dataframe['Year'] < 2018]
    # # test = dataframe[dataframe['Year'] >= 2018]

    # # import the regression module

    # initialize setup
    s = setup(
      data = train, test_data = test, target = 'Count', fold_strategy = 'timeseries', numeric_features = ['Year', 'Series'], fold = 10, transform_target = True, html=False, silent=True, session_id = 123, verbose=False)
    best = compare_models()

    created_model = create_model("lar")

    final_best = finalize_model(created_model)
    start_date = '{}-01-01'.format(end_year + 1)
    end_date = '{}-01-01'.format(end_year + 10)
    future_dates = pd.date_range(start = start_date, end = end_date, freq = 'MS')

    future_df = pd.DataFrame()
    future_df['Month'] = [i.month for i in future_dates]
    future_df['Year'] = [i.year for i in future_dates]    
    future_df['Series'] = np.arange(len(dataframe.index),(len(dataframe.index)+len(future_dates)))

    predictions_future = predict_model(final_best, data=future_df)

    concat_df = pd.concat([dataframe,predictions_future], axis=0)

    concat_df.drop(['Series'], axis=1, inplace=True)
    
    concat_df.drop(['Month'], axis=1, inplace=True)

    concat_df["Count"] = concat_df["Count"].fillna(0)
    concat_df["Label"] = concat_df["Label"].fillna(0)

    concat_df["Count"] = concat_df["Count"].astype(int)
    concat_df["Label"] = concat_df["Label"].astype(int)

    concat_df["Count"] = concat_df["Count"]+concat_df["Label"]
    concat_df.drop(['Label'], axis=1, inplace=True)

    concat_df= concat_df.groupby(["Year"], as_index=False).sum()

    print(concat_df)

    return jsonify({'status': HTTPStatus.OK, 'prediction_data': {
      "values": concat_df["Count"].to_list(),
      "keys":  concat_df["Year"].to_list()
    }, "success": True})

    # fig = px.line(concat_df, x=concat_df.index, y=["Count", "Label"], template = 'plotly_dark')
    # fig.show()
  # except Exception as e:
  #   return jsonify({'status': HTTPStatus.INTERNAL_SERVER_ERROR, 'message': "Not enough data to forcast.", 'exception': str(e.__class__), "success": False})

app.run(debug=True)