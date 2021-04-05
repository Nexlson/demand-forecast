import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import math
from scipy import stats
from torchsummaryX import summary
import time


st.title('Power Usage Forecasting')
st.markdown('----')
st.write(
"""
The increasing demand of energy has raised a huge concern from different parties.
We would like to forecast the future power usage to help regulators, power suppliers or more.

Power usage of a city can be influenced by factors like weather and has a seasonal effect. 
We have tried on different time series models to predict the future power usage based on the power usage in the past.
To read more about our models, [click here](https://github.com/Nexlson/power-usage-forecasting).
"""
)

st.info('The dataset we used is [Smart meters in London](https://www.kaggle.com/jeanmidev/smart-meters-in-london?select=weather_daily_darksky.csv) from Kaggle.')

#load data and show
DATA_URL = ("preprocessed_data.csv")
df = pd.read_csv(DATA_URL)

st.header('Our Cleaned Data')
st.write(df.head(7))

#side panel
#data statistical description
st.sidebar.subheader(' Data Statistics')
st.markdown("Tick the box in side panel to view data statistics.")
if st.sidebar.checkbox('Statistical Description'):
    st.subheader('Statistical Data Descripition')
    st.write(df.describe())

#visualization
st.header('Data Visualization')
df.index = df.date
df.index = pd.to_datetime(df.index)

st.line_chart(df.total_power_consumption)

#Model
st.header('Model')
st.markdown(
"""
We had tried modelling the dataset using SES, Holt's Method, Holt's Winter 
ARIMA, MLP, LSTM (univariate input and single step) and Bi LSTM (multivariate input).

We compare the RMSE loss we have for each model and we decided LSTM is our best model.

Click on 'Build Your Own Model' button to train you own model!
""")

df= None

if st.sidebar.button('Build Your Own Model'):
    #LSTM Model
    st.title('Explore More on LSTM Model')
    st.markdown('Select your setting for your model on the side bar. ')
    st.text(
    """

    Or you can follow our setting:
    Window size = 3
    iddenH dimension = 64
    Number of layers = 1 (vanilla lstm)
    Forecasted step = 1
    Number of epochs = 200

    """)
    st.markdown("To start training, press **'Start Model'**")

    st.sidebar.subheader('LSTM Model')
    window_size = st.sidebar.slider('Window size', 3, 10, 1)
    hidden_dim = st.sidebar.slider('Hidden dimension', 64, 200, 1)
    num_layers = st.sidebar.slider('Number of layers', 1, 10, 1) #num layers :1 for vanila LSTM, >1 is mean stacked LSTM'
    n_step = st.sidebar.slider('Forecasted step', 1, 100, 1)
    num_epochs = st.sidebar.slider('Number of epochs', 200, 1000, 10)

    ##define hyperparameter
    split_ratio = 0.70
    batch_size = 10
    #hidden_dim = 64, window_size = 4

    if st.sidebar.button('Start Model'):
        st.header('Model started.')
        st.write("Setting:")
        st.write("window_size: " + str(window_size) +", hidden_dim: " + str(hidden_dim)+" , num_layers: " + str(num_layers)+" , forecast_step"+str(n_step)+", num_epoch"+str(num_epoch))

        ##data preprocessing
        power = pd.read_csv('preprocessed_data.csv')
        power.drop(['holiday', 'temp_mean'], axis=1, inplace=True)
        power_ts = power.set_index(power['date'])
        power_ts = pd.Series(power_ts['total_power_consumption'])

        split_data_power = round(len(power_ts)*split_ratio)
        train_data_power = power_ts[:split_data_power]
        test_data_power = power_ts[split_data_power:]
        train_time_power = train_data_power.index
        test_time_power = test_data_power.index

        scaler_power = StandardScaler().fit(train_data_power.values.reshape(-1,1))
        scaler_train_power_data = scaler_power.transform(train_data_power.values.reshape(-1,1))
        scaler_test_power_data = scaler_power.transform(test_data_power.values.reshape(-1,1))
        st.text('Data preprocessed.')


        # Data sequencing function
        def univariate_single_step(sequence, window_size):
            x, y = list(), list()
            for i in range(len(sequence)):
            # find the end of this pattern
                end_ix = i + window_size
                # check if we are beyond the sequence
                if end_ix > len(sequence)-1:
                    break
            # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                x.append(seq_x)
                y.append(seq_y)
            return np.array(x), np.array(y)

        ##transform data
        trainX_power, trainY_power = univariate_single_step(scaler_train_power_data, window_size)
        testX_power, testY_power = univariate_single_step(scaler_test_power_data, window_size)
        trainX_power = torch.from_numpy(trainX_power).type(torch.Tensor)
        trainY_power = torch.from_numpy(trainY_power).type(torch.Tensor)
        testX_power = torch.from_numpy(testX_power).type(torch.Tensor)
        testY_power = torch.from_numpy(testY_power).type(torch.Tensor)
        st.text('Data transformed.')

        ##iterator
        train_dataset_power = TensorDataset(trainX_power, trainY_power)
        train_iter_power = DataLoader(train_dataset_power,batch_size = batch_size, shuffle=False)
        test_dataset_power = TensorDataset(testX_power, testY_power)
        test_iter_power = DataLoader(test_dataset_power, batch_size= batch_size, shuffle=False)
        st.text('Make iterator.')

        class LSTM(nn.Module):
            def __init__(self, n_feature, hidden_dim, num_layers, n_step) :
                super(LSTM, self).__init__()
                self.n_feature = n_feature
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.n_step = n_step
                self.lstm = nn.LSTM(n_feature, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, n_step)
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
                
                # We need to detach as we are doing truncated backpropagation through time (BPTT)
                # If we don't, we'll backprop all the way to the start even after going through another batch
                out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
                
                # Index hidden state of last time step
                # we just want last time step hidden states(output)
                out = out[:, -1, :]
                out = self.fc(out)
                
                return out

        torch.manual_seed(123)
        number_of_time_series = 1
        time_step = 1

        # Vanilla or Stacked LSTM
        model_power = LSTM(n_feature=number_of_time_series, hidden_dim = hidden_dim, num_layers=num_layers, n_step =n_step)

        # Define MSE as loss function
        loss_fn_power = torch.nn.MSELoss()

        # Set up optimizer
        optimizer_power = torch.optim.Adam(model_power.parameters(), lr=0.001)

        def training(num_epochs, train_iter, test_iter, optimizer, loss_fn, model):
            # Create a list of zero value to store the averaged value
            train_loss = np.zeros(num_epochs)
            val_loss = np.zeros(num_epochs)
            
            for t in range(num_epochs):
                
                # Initial the value to be zero to perform cummulative sum
                running_loss_train = 0
                running_loss_valid = 0
                
                # For loop to loop through each data in the data iterator
                for _,(train_X, train_Y) in enumerate(train_iter):
                    
                    # Forward pass
                    y_train_pred = model(train_X)
                    
                    # Reshape to ensure the predicted output (y_train_pred) same size with train_Y shape
                    y_train_pred = torch.reshape(y_train_pred, (train_Y.shape[0], train_Y.shape[1]))
                    
                    # Compare the value using MSE
                    loss_train = loss_fn(y_train_pred, train_Y)
                    
                    # Zero out gradient, else they will accumulate between batches 
                    optimizer.zero_grad()
                    
                    # Backward pass
                    loss_train.backward()
                    
                    # Update parameters 
                    optimizer.step()
                    
                    # Since the loss_train.item will only return the average loss based number of batches
                    # loss_train.item()*train_X.size(0)
                    running_loss_train += loss_train.item()*train_X.size(0)
                    
                # Average the loss base on total batch size, train_iter.dataset is use to get total batch size
                epoch_loss_train = running_loss_train / len(train_iter.dataset)
                
                #Store the averaged value
                train_loss[t] = epoch_loss_train
                if t%50 == 0:
                    st.text('Epoch ' + str(t) + ': ' + str(epoch_loss_train))
                
                #Validate the test data loss
                with torch.no_grad():
                    # For loop to loop thorugh each data in the data iterator
                    for j, (test_X, test_Y) in enumerate (test_iter):
                        y_test_pred = model(test_X)
                        
                        # Reshape to ensure the predicted output (y_test_grad) same size with test_y shape
                        y_test_pred = torch.reshape(y_test_pred, (test_Y.shape[0], test_Y.shape[1]))
                        
                        # Calculate the loss
                        loss_test = loss_fn(y_test_pred, test_Y)
                        
                        # Summing up the loss over each batch
                        running_loss_valid += loss_test.item()* test_X.size(0)
                        
                    # Average the loss base on total batch size
                    epoch_loss_test = running_loss_valid /len(test_iter.dataset)
                    
                    # Store the averaged value 
                    val_loss[t] = epoch_loss_test
                    if t%50 == 0:
                        st.text('Epoch ' + str(t) + ': ' + str(epoch_loss_test))
                    
                
            return train_loss, val_loss

        # Start Training
        st.text('Start trainig.')
        train_loss_power , val_loss_power = training(num_epochs, train_iter_power, test_iter_power, optimizer_power, loss_fn_power, model_power)

        #turn off warning
        st.set_option('deprecation.showPyplotGlobalUse', False) 

        #validation
        def learning_curve(num_epochs, train_loss, val_loss):
            fig = plt.figure(figsize=(10,6))
            fig = plt.plot(train_loss, label="Training")
            fig = plt.plot(val_loss, label="Testing")
            fig = plt.xlabel("Epoch")
            fig = plt.ylabel("MSE")
            fig = plt.legend()
            fig = plt.title("Train Loss & Test Loss")
            st.pyplot(figure = fig)
            for i in range(num_epochs):
                print(f'Epoch : {i} , training loss : {train_loss[i]}, validation loss : {val_loss[i]}')

        learning_curve(num_epochs, train_loss_power, val_loss_power)

        # Section 1 : Make predictions
        with torch.no_grad():
            y_train_prediction_power = model_power(trainX_power)
            y_test_prediction_power = model_power(testX_power)
            
        # Section 2 : Reshape to original data    
        y_train_prediction_power = torch.reshape(y_train_prediction_power,(y_train_prediction_power.shape[0], y_train_prediction_power.shape[1]))
        trainY_power = torch.reshape(trainY_power, (trainY_power.shape[0], trainY_power.shape[1]))
        y_test_prediction_power = torch.reshape(y_test_prediction_power,(y_test_prediction_power.shape[0],y_test_prediction_power.shape[1]))
        testY_power = torch.reshape(testY_power,(testY_power.shape[0],testY_power.shape[1]))

        # Section 3 : Invert predictions
        y_train_pred_power = scaler_power.inverse_transform(y_train_prediction_power)
        y_train_power = scaler_power.inverse_transform(trainY_power)
        y_test_pred_power = scaler_power.inverse_transform(y_test_prediction_power)
        y_test_power = scaler_power.inverse_transform(testY_power)

        #for forecast plot in the next section
        train_date = pd.date_range(start='2011-11-01', end='2013-06-23')
        train = pd.DataFrame(train_data_power)
        test_date = pd.to_datetime(pd.date_range(start='2013-06-24', end='2014-02-22', freq='D'),unit = 'D')
        test = pd.DataFrame(y_test_power, index = test_date, columns = ['test']) 
        forecast = pd.DataFrame(y_test_pred_power, index = test_date, columns = ['forecast'])

        df = pd.concat([train, test, forecast], axis=1).reindex(pd.date_range(start='2011-11-01', end='2014-02-22')).reset_index()
        df = df.rename(columns={"total_power_consumption": "test"})

        # Section 4 : Calculate root mean squared error for both train and test data
        trainScore_power = math.sqrt(mean_squared_error(y_train_power, y_train_pred_power))
        st.write('Train Score: %.2f RMSE' % (trainScore_power))
        testScore_power = math.sqrt(mean_squared_error(y_test_power, y_test_pred_power))
        st.write('Test Score: %.2f RMSE' % (testScore_power))

        def single_step_plot(original_test_data,sequence_test_data,forecast_data,test_time,window_size,original_plot =False):
            # Take the time index after data sequence
            sequence_test_time = test_time[window_size:]
            
            plt.figure(figsize=(10,6))
            
            if original_plot:
                plt.plot(test_time,original_test_data,color="blue",label = 'Test Data')
                
            plt.plot(sequence_test_time,sequence_test_data,color="green", label = 'Test Data After Sequence')
            plt.plot(sequence_test_time,forecast_data,color="red", label = 'Forecast')
            plt.xticks(rotation = 45)
            plt.ylabel("Value")
            plt.title("Forecast plot")
            plt.legend()
            st.pyplot()

        single_step_plot(original_test_data = test_data_power,
                        sequence_test_data = y_test_power,
                        forecast_data = y_test_pred_power,
                        test_time = test_time_power,
                        window_size = window_size,
                        original_plot = True )

st.title('Forecast')
import time

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
if df is not None:
    df = df
else: df = pd.read_csv('forecast.csv')


chart = st.line_chart(df[:579], width=1000, height=500)

for i in range(df.shape[0]):
  #  test_rows = plot_test[i*5:i*5+5] 
    forecast_rows =df[579+i*5:579+i*5+5] 
    status_text.text("%i%% Complete" % i)
  #  chart.add_rows(test_rows)
    chart.add_rows(forecast_rows)
    progress_bar.progress(i)
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")