from keras import models
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import timedelta
import datetime as dt
import ta
from sklearn.impute import SimpleImputer
from keras.models import load_model


# Set the Streamlit app title and page icon
st.set_page_config(page_title="Stock Price Prediction", page_icon="📈")

# Custom HTML header
html_temp = """
    <div style="background-color:#8A9A5B;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Stock Price Prediction</h1>
    <h4 style="color:white;text-align:center;">COMPANY : ICICI BANK </h4>
    <h3 style="color:white;text-align:center;">Presented by: Group No 6 </h3>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

# Load the stock price dataset
df = pd.read_csv(r'C:/Users/Naresh/Downloads/Stock predication Projects/ICICIBANK.BO.csv')

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Impute missing values in both 'Open' and 'Close' columns
imputer = SimpleImputer(strategy='mean')
df[['Open', 'Close']] = imputer.fit_transform(df[['Open', 'Close']])

# Display the first 10 rows of the dataset
st.subheader("Stock Price Data:")
st.write(df.head(10))

# Check the column names
st.subheader("Column Names:")
st.write(df.columns)

st.subheader('Data from 2002-2022')
#df= df.reset_index()
st.write(df.tail(10))
st.write(df.describe())


df.columns = [x.lower() for x in df.columns]

st.subheader('Technical Analysis')
infoType = st.radio(
        "Choose Technical Analysis Type",
        ('Moving Average Chart', 'Market trend', 'Williams %R', 'Stochastic Oscillator')
    )
if infoType == 'Moving Average Chart':
    st.subheader('Closing Price vs Time Chart with 100 MA')
    ma100 = df.close.rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100)
    plt.plot(df.close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart with 100 MA & 200MA')
    ma100 = df.close.rolling(100).mean()
    ma200 = df.close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, 'g')
    plt.plot(ma200, 'r')
    plt.plot(df.close, 'b')
    plt.legend()
    st.pyplot(fig)

elif infoType == 'Stochastic Oscillator':
    st.subheader('Stochastic Oscillator')

    # Calculate the MACD and Signal line indicators
    # Calculate the short term exponential moving average
    ShortEMA = df["close"].ewm(span=12, adjust=False).mean()
    # Calculate the long term exponential moving average
    LongEMA = df["close"].ewm(span=26, adjust=False).mean()
    # Calculate the MACD line
    MACD = ShortEMA - LongEMA
    # Calculate the signal line
    signal = MACD.ewm(span=9, adjust=False).mean()
    # Create new columns for the data
    df["MACD"] = MACD
    df["Signal Line"] = signal

    # Find minimum of 14 consecutive values by rolling function
    df['14-low'] = df['low'].rolling(14).min()
    df['14-high'] = df['high'].rolling(14).max()

    # Apply the formula
    df['%K'] = (df['close'] - df['14-low']) * 100 / (df['14-high'] - df['14-low'])
    df['%D'] = df['%K'].rolling(3).mean()

    # Force lowercase (optional)
    df.columns = [x.lower() for x in df.columns]

    # Construct a 2 x 1 Plotly figure
    fig2 = plt.figure(figsize=(16, 10))
    fig2 = make_subplots(rows=2, cols=1)

    # price Line
    fig2.append_trace(go.Scatter(x=df.index, y=df['open'], line=dict(color='#ff9900', width=1),
                                name='open', legendgroup='1',), row=1, col=1)

    # Candlestick chart for pricing
    fig2.append_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'],
                                     close=df['close'], increasing_line_color='#ff9900',
                                     decreasing_line_color='black', showlegend=False), row=1, col=1)

    # Fast Signal (%k)
    fig2.append_trace(go.Scatter(x=df.index, y=df['%k'], line=dict(color='#ff9900', width=2), name='macd',
                                 legendgroup='2',), row=2, col=1)

    # Slow signal (%d)
    fig2.append_trace(go.Scatter(x=df.index, y=df['%d'], line=dict(color='#000000', width=2),
                                 legendgroup='2', name='signal'), row=2, col=1)

    # Colorize the histogram values
    colors = np.where(df['macd'] < 0, '#000', '#ff9900')
    # Plot the histogram
    fig2.append_trace(go.Bar(x=df.index, y=df['macd'], name='histogram', marker_color=colors, ), row=2, col=1)

    # Make it pretty
    layout = go.Layout(autosize=False,
                       width=1000,
                       height=1000, plot_bgcolor='#efefef',
                       font_family='Monospace', font_color='#000000', font_size=20,
                       xaxis=dict(
                           rangeslider=dict(visible=True)))

    # Update options and show plot
    fig2.update_layout(layout)
    st.plotly_chart(fig2)

elif infoType == 'Williams %R':
    st.subheader('Williams %R')

    def get_wr(high, low, close, lookback):
        highh = high.rolling(lookback).max()
        lowl = low.rolling(lookback).min()
        wr = -100 * ((highh - close) / (highh - lowl))
        return wr

    df['wr_14'] = get_wr(df['high'], df['low'], df['close'], 14)

    fig5 = plt.figure(figsize=(15, 12))
    ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((11, 1), (6, 0), rowspan=5, colspan=1)
    ax1.plot(df['close'], linewidth=2)
    ax1.set_title('CLOSING PRICE')
    ax2.plot(df['wr_14'], color='orange', linewidth=2)
    ax2.axhline(-20, linewidth=1.5, linestyle='--', color='grey')
    ax2.axhline(-50, linewidth=1.5, linestyle='--', color='green')
    ax2.axhline(-80, linewidth=1.5, linestyle='--', color='grey')
    ax2.set_title('WILLIAMS %R 14')
    st.pyplot(fig5)

else:
    start = dt.datetime.today() - dt.timedelta(2 * 365)
    end = dt.datetime.today()
    # df = yf.download(user_input, start, end)
    df = df.reset_index()
    fig = go.Figure(
        data=go.Scatter(x=df.index, y=df['adj close'])
    )
    fig.update_layout(
        title={
            'text': "Stock Prices Over Past Two Years",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    st.plotly_chart(fig, use_container_width=True)


data_training = pd.DataFrame(df['close'][0:int(len(df) * 0.70)])

date1 = st.date_input("Enter Date in this format yyyy-mm-dd")
result = st.button("Predict")


# testing part
past_100_days = data_training.tail(100)

# Ensure past_100_days is a DataFrame (remove this line if not needed)
past_100_days = pd.DataFrame(past_100_days)

# Append the DataFrames
final_df = past_100_days.append(data_training, ignore_index=True)

# Create MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale the data
input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)    
y_predicted = models.predict(x_test)
scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# final Graph
st.subheader("Predictions vs Original")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Stock Price Prediction by Date')
# ... (the rest of your code)

st.subheader('Stock Price Prediction by Date')

df1=df.reset_index()['close']
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
#datemax="24/06/2022"
datemax=dt.datetime.strftime(dt.datetime.now() - timedelta(1), "%d/%m/%Y")
datemax =dt.datetime.strptime(datemax,"%d/%m/%Y")
x_input=df1[:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()


date1 = st.date_input("Enter Date in this format yyyy-mm-dd")


if result:
    from datetime import datetime
    my_time = datetime.min.time()
    date1 = datetime.combine(date1, my_time)
    nDay = (datetime.combine(date1, datetime.min.time()) - date1).days  # Keep .days here
    nDay = nDay.days

    date_rng = pd.date_range(start=date1, end=date1, freq='D')
    date_rng = date_rng[1:date_rng.size]
    lst_output = []
    n_steps = x_input.shape[1]
    
    i = 0

    while i <= nDay:
        if len(temp_input) > n_steps:
            x_input = np.array(temp_input[1:])
            print("{} day input {}".format(i, x_input))
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            print("{} day output {}".format(i, yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i = i + 1

    res = scaler.inverse_transform(lst_output)
    output = res[nDay]

    st.write("*Predicted Price for Date :*", date1, "*is*", np.round(output[0], 2))
    st.success('The Price is {}'.format(np.round(output[0], 2)))

    # Corrected indentation for the following lines
    predictions = res[res.size - nDay:res.size]
    print(predictions.shape)
    predictions = predictions.ravel()
    print(type(predictions))
    print(date_rng)
    print(predictions)
    print(date_rng.shape)

    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    df = pd.DataFrame(data=date_rng)
    df['Predictions'] = predictions.tolist()
    df.columns = ['Date', 'Price']
    st.write(df)
    csv = convert_df(df)
    st.download_button(
        "Press to Download",
        csv,
        "file.csv",
        "text/csv",
        key='download-csv'
    )
    # visualization

    fig = plt.figure(figsize=(10, 6))
    xpoints = date_rng
    ypoints = predictions
    plt.xticks(rotation=90)
    plt.plot(xpoints, ypoints)
    st.pyplot(fig)


# scaling of data using min max scaler (0,1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)
# Line chart for actual and predicted prices over time
st.subheader("Actual vs Predicted Prices Over Time:")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['date'], df['close'], label='Actual Prices', marker='o')
ax.plot(df['Date'], model.predict(df[['Open']]), label='Predicted Prices', marker='x')

# Extend the date range for the next 30 days
future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), periods=30, freq='D')
future_dates_df = pd.DataFrame({'Date': future_dates, 'Open': np.nan})

# Predict the Close prices for the next 30 days
future_close_predictions = model.predict(future_dates_df[['Open']])
future_dates_df['Close'] = future_close_predictions

# Convert 'Date' column to string
future_dates_df['Date'] = future_dates_df['Date'].astype(str)

ax.plot(future_dates_df['Date'], future_dates_df['Close'], label='Future Predictions', linestyle='--', marker='o')

ax.set_title('Actual vs Predicted Prices Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# Calendar for date selection
st.subheader("Select a Date:")
min_date = pd.to_datetime(df['Date'].min())
max_date = pd.to_datetime(df['Date'].max())
default_date = min_date

selected_date = st.date_input("Pick a date", min_value=min_date, max_value=max_date, value=default_date)

# Filter data based on the selected date
selected_data = df[df['Date'] == selected_date]

# Display the selected data
st.subheader("Selected Data:")
st.write(selected_data)

# User Input for Prediction
st.subheader("Predict Stock Price:")
open_price = st.number_input("Enter the Open Price:", min_value=float(X.min()), max_value=float(X.max()))
predicted_close_price = model.predict([[open_price]])

# Display the predicted close price
st.write(f"Predicted Close Price: {predicted_close_price[0]:.2f}")

# Display future predictions for the next 30 days
st.subheader("Future Predictions for the Next 30 Days:")
st.write(future_dates_df)

# Additional text
st.text("Feel free to explore and analyze the stock price data!")

# Note: This is a basic example, and you might want to use more sophisticated models for accurate predictions.
