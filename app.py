import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('/Stock_Market_Prediction_ML (4)/Gold-price-prediction.keras')

st.set_page_config(layout="wide")
st.title('Stock Market Predictor')

# Section: Get stock data
st.sidebar.header('Select Stock Symbol')
stock_symbol = st.sidebar.text_input('Enter Stock Symbol', 'XAUT-USD')


stock = stock_symbol

start = '2012-01-01'
end = datetime.today().strftime('%Y-%m-%d')
data = yf.download(stock_symbol, start, end)
data.reset_index(inplace=True)

# Section: Display Stock Data
st.header(f'{stock} ({stock_symbol}) Stock Price & Analysis')

# Current Price Section
current_price = data['Close'].iloc[-1]
previous_close = data['Close'].iloc[-2]
change = current_price - previous_close
percentage_change = (change / previous_close) * 100

st.metric(label="Current Price", value=f"${current_price:.2f}", delta=f"{change:.2f} ({percentage_change:.2f}%)")

st.write(data)

# # Add color to the volume bars
# volume_colors = ['green' if data['Close'][i] > data['Close'][i-1] else 'red' for i in range(1, len(data))]
# volume_colors.insert(0, 'grey')  # First day has no previous day to compare

# # Plotly chart for time series data
# fig = go.Figure()

# fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
# fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume', marker_color=volume_colors, yaxis='y2', opacity=0.3))

# fig.update_layout(
#     title=f"{stock} Stock Chart & Stats",
#     yaxis=dict(title='Price'),
#     yaxis2=dict(title='Volume', overlaying='y', side='right'),
#     xaxis=dict(title='Date'),
#     legend=dict(x=0, y=1.0),
#     margin=dict(l=0, r=0, t=40, b=0),
#     hovermode='x',
# )

# fig.update_traces(
#     hovertemplate="<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<br><b>Volume</b>: %{y2}<extra></extra>",
#     hoverlabel=dict(bgcolor="white")
# )

# # Display chart
# st.plotly_chart(fig, use_container_width=True)



# Section: Advanced Chart
st.subheader("Advanced Chart")

tab_names = ["1d", "1w", "1m", "3m", "6m", "YTD", "1y", "3y", "5y", "10y"]
tabs = st.tabs(tab_names)

for i, tab in enumerate(tabs):
    with tab:
        if tab_names[i] == "1d":
            # Use minute-level data for one day
            start_day = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
            end_day = datetime.now().strftime('%Y-%m-%d')
            day_data = yf.download(stock_symbol, start=start_day, end=end_day, interval='1m')
            day_data.reset_index(inplace=True)

            if not day_data.empty:
                fig = go.Figure()

                # Add price line with conditional color
                fig.add_trace(go.Scatter(
                    x=day_data.index, y=day_data['Close'], mode='lines',
                    line=dict(color='green' if day_data['Close'].iloc[-1] > day_data['Open'].iloc[-1] else 'red'),
                    name='Price',
                    hoverinfo='x+y'
                ))

                # Add volume bars
                colors = ['green' if row['Close'] > row['Open'] else 'red' for index, row in day_data.iterrows()]
                fig.add_trace(go.Bar(
                    x=day_data.index, y=day_data['Volume'], name='Volume',
                    marker_color=colors, yaxis='y2', opacity=0.3
                ))

                fig.update_layout(
                    title='1 Day Price',
                    yaxis=dict(title='Price'),
                    yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
                    xaxis=dict(title='Time', tickformat='%H:%M'),
                    legend=dict(x=0, y=1.0),
                    margin=dict(l=0, r=0, t=40, b=0),
                    hovermode='x',
                    dragmode=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No data available for the selected period.")
        else:
            date_interval_map = {
                "1w": 7,
                "1m": 30,
                "3m": 90,
                "6m": 180,
                "YTD": 365,
                "1y": 365,
                "3y": 3 * 365,
                "5y": 5 * 365,
                "10y": 10 * 365,
            }
            interval = date_interval_map.get(tab_names[i])
            data_interval = data.tail(interval)
            
            if not data_interval.empty:
                fig = go.Figure()

                # Add price line with conditional color
                fig.add_trace(go.Scatter(
                    x=data_interval['Date'], y=data_interval['Close'], mode='lines',
                    line=dict(color='green' if data_interval['Close'].iloc[-1] > data_interval['Open'].iloc[-1] else 'red'),
                    name='Price',
                    hoverinfo='x+y'
                ))

                # Add volume bars
                colors = ['green' if row['Close'] > row['Open'] else 'red' for index, row in data_interval.iterrows()]
                fig.add_trace(go.Bar(
                    x=data_interval['Date'], y=data_interval['Volume'], name='Volume',
                    marker_color=colors, yaxis='y2', opacity=0.3
                ))

                fig.update_layout(
                    title=f'{tab_names[i]} Price',
                    yaxis=dict(title='Price'),
                    yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
                    xaxis=dict(title='Date'),
                    legend=dict(x=0, y=1.0),
                    margin=dict(l=0, r=0, t=40, b=0),
                    hovermode='x',
                    dragmode=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No data available for the selected period.")


# Section: Plot moving averages
with st.container():
    st.header('Moving Averages')
    tab1, tab2, tab3 = st.tabs(["MA50", "MA50 vs MA100", "MA100 vs MA200"])

    with tab1:
        st.subheader('Price vs MA50')
        ma_50_days = data.Close.rolling(50).mean()
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data['Date'], y=ma_50_days, mode='lines', name='MA50'))
        fig1.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.subheader('Price vs MA50 vs MA100')
        ma_100_days = data.Close.rolling(100).mean()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data['Date'], y=ma_50_days, mode='lines', name='MA50'))
        fig2.add_trace(go.Scatter(x=data['Date'], y=ma_100_days, mode='lines', name='MA100'))
        fig2.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader('Price vs MA100 vs MA200')
        ma_200_days = data.Close.rolling(200).mean()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=data['Date'], y=ma_100_days, mode='lines', name='MA100'))
        fig3.add_trace(go.Scatter(x=data['Date'], y=ma_200_days, mode='lines', name='MA200'))
        fig3.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
        st.plotly_chart(fig3, use_container_width=True)


# Section: Predict the next 30 days
window_size = 60
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['Close']])

x_test = []
x_test.append(data_scaled[-window_size:])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = []
for i in range(30):
    predicted_price_scaled = model.predict(x_test).reshape(-1, 1)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    predicted_prices.append(predicted_price[0, 0])
    
    new_scaled_value = predicted_price_scaled[0, 0].reshape(-1, 1, 1)
    x_test = np.append(x_test[:, 1:, :], new_scaled_value, axis=1)

predicted_prices = np.array(predicted_prices)

# Section: Plot the historical data and the prediction
st.subheader(f'Predicted {stock} Prices for the Next 30 Days')

# Create a Plotly figure for historical data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Historical Price', line=dict(color='blue')))

# Add the predicted prices to the Plotly figure
last_date = data['Date'].iloc[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices, mode='lines', name='Predicted Prices', line=dict(color='red')))

fig.update_layout(title=f'{stock} Price History and Predicted Prices for the Next 30 Days',
                    xaxis_title='Date', yaxis_title='Price',
                    hovermode='x unified')

st.plotly_chart(fig, use_container_width=True)

zoomed_data = data.tail(100)
# Create a separate Plotly figure for the zoomed-in plot
fig_zoom = go.Figure()
fig_zoom.add_trace(go.Scatter(x=zoomed_data['Date'], y=zoomed_data['Close'], mode='lines', name='Historical Price', line=dict(color='blue')))
fig_zoom.add_trace(go.Scatter(x=future_dates, y=predicted_prices, mode='lines', name='Predicted Prices', line=dict(color='red')))

fig_zoom.update_layout(title=f'{stock} Zoomed Price History and Predicted Prices for the Next 30 Days',
                        xaxis_title='Date', yaxis_title='Price',
                        hovermode='x unified')

st.plotly_chart(fig_zoom, use_container_width=True)


# Display predicted prices in a scrollable table
days = [f'Day {i+1}' for i in range(len(predicted_prices))]
predicted_data = {'Day': days, 'Predicted Price (USD)': [f'${price:.2f}' for price in predicted_prices]}
predicted_df = pd.DataFrame(predicted_data)

st.write("### Predicted Prices:")
st.dataframe(predicted_df)