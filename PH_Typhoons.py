import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time

if "data_norm" not in st.session_state:
    st.session_state.data_norm = None

if "model" not in st.session_state:
    st.session_state.model = None

def app():
    st.subheader('RNN-LSTM Based Typhoon Prediction in the Philippines')
    
    text = """Shina R. Guazon BSCS 3B AI
    \nCCS 229 - Intelligent Systems
    *Department of Computer Science
    *College of Information and Communications Technology
    *##West Visayas State University##"""
    st.text(text)

    text = """This Streamlit app utilizes a bi-directional Recurrent Neural Network 
    (RNN) with Long Short-Term Memory (LSTM) units to analyze historical typhoon 
    data and forecast the likelihood of typhoons affecting the Philippines in a 
    given month. Users can interact with the app to visualize past typhoon 
    patterns and receive monthly forecasts, potentially aiding in disaster 
    preparedness efforts."""
    st.write(text)

    text = """The data is obtained from the following site : 
    https://en.wikipedia.org/wiki/List_of_typhoons_in_the_Philippines_(2000%E2%80%93present)"""
    st.write(text)  

    df = pd.read_csv('./guazon_ph_typhoons.csv', header=0)

    with st.expander('View Dataset'):
        # Load the data
        st.write(df)    
        st.write(df.shape)

    #re-index and use the Month column data convert to numpy datatime datatype
    df.index = np.array(df['Month'], dtype='datetime64')
    time_axis = df.index

    # Plot the data
    fig, ax = plt.subplots()
    ax.plot(df['Typhoons']) 
    ax.set_xlabel("Time")
    ax.set_ylabel("No. of Typhoons")
    ax.set_title("Time Series Plot of Typhoons")
    st.pyplot(fig)    

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_norm = scaler.fit_transform(df.iloc[:,1].values.reshape(-1, 1))
    data_norm = pd.DataFrame(data_norm)
    st.session_state.data_norm = data_norm

    # Split the data into training and testing sets
    train_size = int(len(data_norm) * 0.8)
    test_size = len(data_norm) - train_size
    train_data, test_data = data_norm.iloc[0:train_size], data_norm.iloc[train_size:len(data_norm)]

    # Convert the data to numpy arrays
    x_train, y_train = train_data.iloc[:-1], train_data.iloc[1:]
    x_test, y_test = test_data.iloc[:-1], test_data.iloc[1:]

    # Reshape the data to match the input shape of the LSTM model
    x_train = np.reshape(x_train.to_numpy(), (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test.to_numpy(), (x_test.shape[0], 1, x_test.shape[1]))
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    look_back = 12  # Number of past days to consider
    n_features = 1  # Number of features in your typhoon data

    model =  tf.keras.Sequential([  # Use Bidirectional LSTM or GRU (comment out the other)
        #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), input_shape=(look_back, n_features)),
        tf.keras.layers.GRU(128, return_sequences=True, input_shape=(look_back, n_features)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.GRU(64, return_sequences=True),  # Another GRU layer
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(32),  # Reduced units for final layer
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1)
    ])

    # Compile the model
    model.compile(loss="mse", optimizer="adam")  # You can adjust loss and optimizer based on your needs

    # Print model summary
    model.summary()
    
    if st.sidebar.button("Start Training"):
        progress_bar = st.progress(0, text="Training the LSTM network, please wait...")           
        # Train the model
        history = model.fit(x_train, y_train, epochs=200, batch_size=64, validation_data=(x_test, y_test))

        fig, ax = plt.subplots()  # Create a figure and an axes
        ax.plot(history.history['loss'], label='Train')  # Plot training loss on ax
        ax.plot(history.history['val_loss'], label='Validation')  # Plot validation loss on ax

        ax.set_title('Model loss')  # Set title on ax
        ax.set_ylabel('Loss')  # Set y-label on ax
        ax.set_xlabel('Epoch')  # Set x-label on ax

        ax.legend()  # Add legend
        st.pyplot(fig)
        st.session_state.model = model

        # update the progress bar
        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)
        # Progress bar reaches 100% after the loop completes
        st.success("LSTM Network training completed!") 

    years = st.sidebar.slider(   
        label="Set the number years to project:",
        min_value=2,
        max_value=6,
        value=2,
        step=1
    )


    if st.sidebar.button("Predictions"):
        # Get the predicted values and compute the accuracy metrics
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        st.write('Train RMSE:', train_rmse)
        st.write('Test RMSE:', test_rmse)
        st.write('Train MAE:', train_mae)
        st.write('Test MAE:', test_mae)

        model = st.session_state.model
        data_norm = st.session_state.data_norm
        # Get predicted data from the model using the normalized values
        predictions = model.predict(data_norm)

        # Inverse transform the predictions to get the original scale
        predvalues = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        predvalues = pd.DataFrame(predvalues)

        # Convert column 'Typhoons' to integer
        predvalues[0] = predvalues[0].astype(int)

        pred_period = years * 12    
        # Use the model to predict the next year of data
        input_seq_len = 24         
        num_features=1
        last_seq = data_norm[-input_seq_len:] # Use the last year of training data as the starting sequence

        preds = []
        for i in range(pred_period):
            pred = model.predict(last_seq)
            preds.append(pred[0])

            last_seq = np.array(last_seq)
            last_seq = np.vstack((last_seq[1:], pred[0]))
            last_seq = pd.DataFrame(last_seq)

        # Inverse transform the predictions to get the original scale
        prednext = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

        #flatten the array from 2-dim to 1-dim
        prednext = [item for sublist in prednext for item in sublist]


        # Generate an array of datetime64 objects from January 1976 to December 1976
        if pred_period == 12:
            end = '2023-12'
        elif pred_period == 24:
            end = '2024-12' 
        elif pred_period == 36:
            end = '2025-12'
        elif pred_period == 48:
            end = '2026-12'
        elif pred_period == 60:
            end = '2027-12'
        elif pred_period == 72:
            end = '2028-12'

        months = pd.date_range(start='2023-01', end=end, freq='MS')

        # Create a Pandas DataFrame with the datetime and values columns
        nextyear = pd.DataFrame({'Month': months, 'Typhoons': prednext})

        # Convert column 'Typhoons' to integer
        nextyear['Typhoons'] = nextyear['Typhoons'].astype(int)

        time_axis = np.linspace(0, df.shape[0]-1, pred_period)
        time_axis = np.array([int(i) for i in time_axis])
        time_axisLabels = np.array(df.index, dtype='datetime64[D]')

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 2.1, 2])

        # make the prediction plot resize based on the years
        ax1 = fig.add_axes([2.3, 0, 0.15 * years, 2])

        ax.set_title('Comparison of Actual and Predicted Data')
        ax.plot(df.iloc[:,1].values, label='Original Dataset')
        ax.plot(list(predvalues[0]), color='red', label='Model  Predictions')

        # Get the maximum y-value among both datasets
        max_y_value = max(df.iloc[:,1].values.max(), nextyear['Typhoons'].max())+2
        # Set the same y-limits for both axes
        ax.set_ylim(0, max_y_value)
        ax1.set_ylim(0, max_y_value)

        ax.set_xticks(time_axis)
        ax.set_xlabel('\nMonth', fontsize=20, fontweight='bold')
        ax.set_ylabel('No. of Typhoons', fontsize=20, fontweight='bold')
        ax.set_xticklabels(time_axisLabels[time_axis], rotation=45)        
        ax.legend(loc='best', prop={'size':20})
        ax.tick_params(size=10, labelsize=15)

        ax1.set_title('Projected No. of Typhoons')
        ax1.plot(nextyear['Typhoons'], color='red', label='predicted next year')
        ax1.set_xlabel('Month', fontsize=20, fontweight='bold')
        ax1.set_ylabel('No. of Typhoons', fontsize=20, fontweight='bold')
        ax1.set_xticklabels(np.array(nextyear['Month'], dtype='datetime64[D]'), rotation=45)        
        ax1.tick_params(size=10, labelsize=15)

        st.pyplot(fig)  

        st.write('Predicted Typhoons for the next', years, 'years:')
        st.write(nextyear)

if __name__ == '__main__':
    app()   
