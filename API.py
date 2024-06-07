import pandas as pd
from flask import Flask, request
from flask_restful import Api, Resource
from sqlalchemy import create_engine
from ML_Functions import RF_Forecast, LSTM_Forecast


app = Flask(__name__)
api = Api(app)


class Forecast(Resource):
    def post(self):
        # Load data from request
        data = request.json

        # Extract parameters
        user = data['user']
        password = data['password']
        database = data['database']
        training_period = data['training_period']
        testing_period = data['testing_period']
        number_val_months = data['number_val_months']
        batch_size = data['batch_size']
        save_folder = data['save_folder']
        model = data['model']


        # Read and convert the csv
        DB_Address = f'mysql+mysqlconnector://{user}:{password}@localhost/{database}'
        engine = create_engine(DB_Address)
        df = pd.read_sql('SELECT * FROM mytable', engine)
        df.rename(columns={'Date': 'index'}, inplace=True)
        df['index'] = pd.to_datetime(df['index'])

        # Call the forecasting function

        if model == "RF":
            df_all_yhat, df_optimal_parameters = RF_Forecast(data=df,
                                                             training_period=training_period,
                                                             testing_period=testing_period,
                                                             number_val_months=number_val_months,
                                                             save_folder=save_folder)
        elif model == "LSTM":
            df_all_yhat, df_optimal_parameters = LSTM_Forecast(data=df,
                                                               training_period=training_period,
                                                               testing_period=testing_period,
                                                               number_val_months=number_val_months,
                                                               batch_size=batch_size,
                                                               save_folder=save_folder)


api.add_resource(Forecast, '/forecast')

if __name__ == "__main__":
    app.run(debug=True)
