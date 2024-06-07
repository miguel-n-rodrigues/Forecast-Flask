import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from ML_Functions import RF_Forecast, LSTM_Forecast

# Load and read table
engine = create_engine('mysql+mysqlconnector://root:*o4m4z%x7u$7BABn@localhost/mydatabase')
df = pd.read_sql('SELECT * FROM mytable', engine)
#df = pd.read_csv("WaterData.csv")
df.rename(columns={'Date': 'index'}, inplace=True)
df['index'] = pd.to_datetime(df['index'])

print(df.describe())


# EDA
df_plot = df.copy()
df_plot.rename(columns={'index': 'Date'}, inplace=True)
df_plot['Date'] = pd.to_datetime(df_plot['Date'])

plt.figure(figsize=(12, 8))
plt.plot(df_plot['Date'], df_plot['WU_SFH_Num'], marker='o', linestyle='-', label='WU_SFH_Num')
plt.plot(df_plot['Date'], df_plot['WU_MFH_Num'], marker='s', linestyle='-', label='WU_MFH_Num')
plt.plot(df_plot['Date'], df_plot['WU_Com_Num'], marker='^', linestyle='-', label='WU_Com_Num')
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Variable Evolution over Time')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(12, 8))
plt.plot(df_plot['Date'], df_plot['WU_SFH_Gal'], marker='o', linestyle='-', label='WU_SFH_Num')
plt.plot(df_plot['Date'], df_plot['WU_MFH_Gal'], marker='s', linestyle='-', label='WU_MFH_Num')
plt.plot(df_plot['Date'], df_plot['WU_Com_Gal'], marker='^', linestyle='-', label='WU_Com_Num')
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Variable Evolution over Time')
plt.legend()
plt.grid(True)
plt.show()


# Forecasts

RF_Forecast(data=df, training_period=72, testing_period=6, number_val_months=4, save_folder="data")

LSTM_Forecast(data=df, training_period=66, testing_period=6, number_val_months=4, batch_size=6, save_folder="data")



