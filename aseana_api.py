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

    dataframe['Year'] = [i.year for i in dataframe['DateTime']]
    dataframe= dataframe.groupby(["Year"], as_index=False).sum()
    # dataframe['MA12'] = dataframe['Count'].rolling(3).mean()

    for i in range(10):
        new_value = (dataframe["Count"].iloc[-1] + dataframe["Count"].iloc[-2] + dataframe["Count"].iloc[-3])/3
        new_row = {'Year': int(dataframe["Year"].iloc[-1]) + 1, 'Count':new_value}
        dataframe = dataframe.append(new_row, ignore_index=True)

    dataframe["Year"] = dataframe["Year"].astype(int)
    dataframe["Count"] = dataframe["Count"].astype(int)

    return jsonify({'status': HTTPStatus.OK, 'prediction_data': {
      "values": dataframe["Count"].to_list(),
      "keys":  dataframe["Year"].to_list()
    }, "success": True})

    # fig = px.line(concat_df, x=concat_df.index, y=["Count", "Label"], template = 'plotly_dark')
    # fig.show()
  except Exception as e:
    return jsonify({'status': HTTPStatus.INTERNAL_SERVER_ERROR, 'message': "Not enough data to forcast.", 'exception': str(e.__class__), "success": False})

app.run(debug=True)