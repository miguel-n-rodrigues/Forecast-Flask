# Introduction

I've developed an application capable of forecasting future values by using both a Random Forest (RF) and Long-Short Term Memory Neural Network (LSTM) approach.

This application is flexible in the size of the data that can be used, although it only accepts values and no character-related variables.

## Data used

The data used in this example was taken from Google Dataset Search (https://datasetsearch.research.google.com/), more specifically a Dataset related to water usage estimation for Rhode Island.

The dataset used is from June 2014 to June 2021, with a monthly granularity, and can be found at (https://datasetsearch.research.google.com/search?src=0&query=demand%20regression&docid=L2cvMTF2cm04MDg1dw%3D%3D) and (https://www.sciencebase.gov/catalog/item/62fea1c9d34e3a444287614d).





# Implementation

## Set up a database and load the data.

The dataset was uploaded to a database server using MySQL.

## Build EDA and Model

Python script responsible for reading the dataset, extracting relevant information, and performing the necessary forecasts was realized. The first row of data was dropped since it had missing values and facilitated future biannual forecasts.

First, by looking at the mean, min, and max values it's possible to understand that there were no abnormal / outliers in the data.

Second, with the aid of graphs, it is also apparent that some variables hold strong correlations between themselves.

![image](https://github.com/miguel-n-rodrigues/DareData-TechChallenge/assets/11060792/a4e693a5-aa9a-4e78-8971-f5bd365f42ed)

![image](https://github.com/miguel-n-rodrigues/DareData-TechChallenge/assets/11060792/7ad897d4-628b-48c7-8bd3-e14db6a5c28a)

Third, the necessary functions were developed in order to take the data and provide the next observation (next month) forecasts by using both RF and LSTM's. These functions are robust with different arguments that can be changed by the used, such as:

* data: the data frame that will be fed to the models
* training_period: the number of observations that will be used to train the models
* testing_period: the number of observations that the models will use to forecast into the future to test their performance
* number_val_months: number of sets of testing_period observations that will be used as validation to find the best set of (hyper)parameters for the models
* batch_size (only for the LSTM model): the size of the batch used for the neural network
* save_folder: specify the location where the forecasts and the optimal (hyper)parameters used to achieve the forecasts should be stored



## Deploy Model

The aforementioned models were then converted to be used with an API, specifically via the Flask framework.

Through the API it is possible to:

* Define the SQL address where the data is stored using the fields 'user', 'password', and 'database'
* Choose the model that we want to use (LSTM or RF)
* Outline the parameters we want to use in those models

For this case study, the forecasts produced were made from January 2020 to June 2021, employing both models and using the following values:

* training_period: 66
* testing_period: 6
* number_val_months: 4
* batch_size (only for the LSTM model): 6
* save_folder: data



# Results

The models print a progression status throughout its execution as seen below:

![image](https://github.com/miguel-n-rodrigues/DareData-TechChallenge/assets/11060792/57874c59-08f3-4e2e-8227-7c67ad77a2f8)

![image](https://github.com/miguel-n-rodrigues/DareData-TechChallenge/assets/11060792/1922076c-4849-4ff1-acb1-eec55dfa0365)


Overall, the models had satisfactory results with the RF forecasts having an RMSE and MAPE of 4000.74 AND 3.63 %, while the LSTM forecasts had an RMSE and MAPE of 5394.78 and 5.29 %.



# Future work

The application can be further worked on to provide better and more user-friendly uses.

These include:

* Have more parameters that can be specified while also making it so that some of these have a default value when they are not, allowing the user more control but not forcing them to do it
* Transform this into a container application for easier deployment
* Allowing the user to save the produced forecasts
* Produce better outputs, including graphs for easier observation
* Add more models that can be used
