import requests

url = 'http://localhost:5000/forecast'

# Prepare the data to send
data = {
    'user': 'root',
    'password': '*o4m4z%x7u$7BABn',
    'database': 'mydatabase',
    'training_period': 78,
    'testing_period': 6,
    'number_val_months': 4,
    'batch_size': 6,
    'save_folder': "data",
    'model': "RF"
}

# Send the request
response = requests.post(url, json=data)
