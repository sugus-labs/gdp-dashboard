import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import plotly.express as px
from datetime import timedelta
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid

# Set Streamlit page configuration
st.set_page_config(page_title="ARMADA Sales Forecasting", layout="wide")

def load_data(file_path):
    """
    Load sales data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        st.error("The specified file was not found.")
        return None


@st.cache_data
def preprocess_data(data, date_col, target_col):
    """
    Preprocess the data for Prophet model.
    """
    df = data[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    df.dropna(inplace=True)

    # Remove zero and negative values
    df = df[df['y'] > 0]

    # Outlier detection and removal using IQR
    Q1 = df['y'].quantile(0.25)
    Q3 = df['y'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = 1.5 * IQR
    df = df[(df['y'] >= Q1 - outlier_threshold) & (df['y'] <= Q3 + outlier_threshold)]

    # Log transformation to stabilize variance
    df['y'] = np.log(df['y'])

    # Add additional regressors
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month

    return df

def add_holidays():
    """
    Define custom holidays.
    """
    holidays = pd.DataFrame({
        'holiday': [
            'Black Friday', 'New Year', 'Ramadan', 'Eid al-Fitr', 'Eid al-Adha',
            'Kuwait National Day', 'Kuwait Liberation Day'
        ] * 3,
        'ds': pd.to_datetime([
            # 2022
            '2022-11-25', '2022-01-01', '2022-04-02', '2022-05-02',
            '2022-07-09', '2022-02-25', '2022-02-26',
            # 2023
            '2023-11-24', '2023-01-01', '2023-03-23', '2023-04-21',
            '2023-06-28', '2023-02-25', '2023-02-26',
            # 2024
            '2024-11-29', '2024-01-01', '2024-03-10', '2024-04-09',
            '2024-06-16', '2024-02-25', '2024-02-26'
        ]),
        'lower_window': [0, 0, 0, -5, -21, 0, 0] * 3,
        'upper_window': [0, 0, 29, 5, 1, 0, 0] * 3
    })
    return holidays

def train_model(train_data, holidays, changepoint_prior, seasonality_prior, n_changepoints):
    """
    Initialize and train the Prophet model.
    """
    model = Prophet(
        changepoint_prior_scale=changepoint_prior,
        seasonality_prior_scale=seasonality_prior,
        holidays=holidays,
        n_changepoints=n_changepoints,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    # Adding custom seasonalities
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)

    # Adding regressors
    model.add_regressor('day_of_week')
    model.add_regressor('month')

    with st.spinner("Training the model..."):
        model.fit(train_data)
    return model


def make_forecast(model, periods, include_history=True):
    """
    Make future predictions using the trained model.
    """
    future = model.make_future_dataframe(periods=periods, include_history=include_history)
    future['day_of_week'] = future['ds'].dt.dayofweek
    future['month'] = future['ds'].dt.month
    forecast = model.predict(future)
    return forecast

def evaluate_model(model, test_data):
    """
    Evaluate the model using test data.
    """
    # Make predictions
    future_test = model.make_future_dataframe(periods=len(test_data), include_history=False)
    future_test['day_of_week'] = future_test['ds'].dt.dayofweek
    future_test['month'] = future_test['ds'].dt.month
    forecast = model.predict(future_test)
    predicted = forecast[['ds', 'yhat']].set_index('ds')
    actual = test_data[['ds', 'y']].set_index('ds')
    results = actual.join(predicted, how='inner')

    # Exponentiate to reverse log transformation
    results['y'] = np.exp(results['y'])
    results['yhat'] = np.exp(results['yhat'])

    mape = mean_absolute_percentage_error(results['y'], results['yhat']) * 100
    mae = mean_absolute_error(results['y'], results['yhat'])
    rmse = mean_squared_error(results['y'], results['yhat'], squared=False)

    return {
        'mape': mape,
        'mae': mae,
        'rmse': rmse,
        'forecast': forecast,
        'results': results
    }


def hyperparameter_tuning(train_data, test_data, holidays):
    """
    Perform hyperparameter tuning using grid search.
    """
    param_grid = {  
        'changepoint_prior_scale': [0.001, 0.01, 0.1],
        'seasonality_prior_scale': [1.0, 5.0, 10.0],
        'n_changepoints': [25, 50, 100]
    }

    best_mape = float('inf')
    best_params = None
    best_model = None

    for params in ParameterGrid(param_grid):
        model = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            holidays=holidays,
            n_changepoints=params['n_changepoints'],
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        # Adding custom seasonalities
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
        # Adding regressors
        model.add_regressor('day_of_week')
        model.add_regressor('month')
        model.fit(train_data)

        # Evaluate model
        evaluation = evaluate_model(model, test_data)
        mape = evaluation['mape']

        if mape < best_mape:
            best_mape = mape
            best_params = params
            best_model = model

    st.write(f"Best MAPE after hyperparameter tuning: {best_mape:.2f}%")
    st.write(f"Optimal parameters: {best_params}")
    return best_model, best_params


def plot_forecast(merged_df):
    """
    Plot the actual and forecasted sales.
    """
    fig = px.line(
        merged_df,
        x='DATE',
        y=['REAL SALES', 'FORECASTED SALES'],
        labels={'value': 'Sales', 'variable': 'Legend'},
        title="Actual vs Forecasted Sales"
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Sales Forecasting for ARMADA")

    # File upload
    #st.subheader("Upload your time series data")
    #uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    uploaded_file = 'data/invoices-headers-v1.csv'  

    if uploaded_file is not None:
        # Load data
        sales_df = load_data(uploaded_file)
        if sales_df is not None:
            #st.write("Data Preview:")
            #st.dataframe(sales_df.head())

            # Select date and target columns
            #date_col = st.selectbox("Select Date Column", sales_df.columns)
            #target_col = st.selectbox("Select Target Column (Sales Amount)", sales_df.columns)
            date_col = 'DATE'
            target_col = 'AMOUNT'

            # Convert date column to datetime and remove timezone
            sales_df[date_col] = pd.to_datetime(sales_df[date_col]).dt.tz_localize(None)

            # Preprocess data
            prophet_data = preprocess_data(sales_df, date_col, target_col)

            # Check if data is sufficient
            if len(prophet_data) < 30:
                st.error("Not enough data points. Please provide more data.")
                return

            # Train-Test Split
            test_size = st.sidebar.slider("Select the number of periods for testing", min_value=1, max_value=365, value=12)
            test_size = 12
            train_data = prophet_data.iloc[:-test_size]
            test_data = prophet_data.iloc[-test_size:]

            # Include holidays
            include_holidays = st.sidebar.checkbox("Include Custom Holidays", value=True)
            holidays = add_holidays() if include_holidays else None

            # Hyperparameter tuning
            st.sidebar.subheader("Hyperparameter Tuning")
            if st.sidebar.button("Run Hyperparameter Tuning"):
                best_model, best_params = hyperparameter_tuning(train_data, test_data, holidays)
            else:
                # Default parameters
                changepoint_prior = st.sidebar.slider(
                    "Changepoint Prior Scale", min_value=0.001, max_value=0.5, value=0.05, step=0.001
                )
                seasonality_prior = st.sidebar.slider(
                    "Seasonality Prior Scale", min_value=0.01, max_value=10.0, value=5.0, step=0.01
                )
                n_changepoints = st.sidebar.selectbox(
                    "Number of Changepoints", options=[25, 50, 100], index=1
                )
                # Train model
                best_model = train_model(train_data, holidays, changepoint_prior, seasonality_prior, n_changepoints)

            # Make forecast
            periods_input = st.sidebar.number_input(
                "How many periods into the future would you like to forecast?", min_value=1, max_value=365, value=365
            )
            forecast = make_forecast(best_model, periods_input)

            # Exponentiate to reverse log transformation
            forecast['yhat'] = np.exp(forecast['yhat'])
            forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
            forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])

            # Prepare data for visualization
            forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            forecast_df.rename(columns={
                'ds': 'DATE',
                'yhat': 'FORECASTED SALES',
                'yhat_lower': 'Lower Confidence Interval',
                'yhat_upper': 'Upper Confidence Interval'
            }, inplace=True)

            sales_df.rename(columns={target_col: 'REAL SALES', date_col: 'DATE'}, inplace=True)

            # Ensure 'DATE' columns are datetime and timezone-naive
            sales_df['DATE'] = pd.to_datetime(sales_df['DATE']).dt.tz_localize(None)
            forecast_df['DATE'] = pd.to_datetime(forecast_df['DATE']).dt.tz_localize(None)

            # Merge actual and forecasted data
            merged_df = pd.merge(sales_df, forecast_df, on='DATE', how='outer')
            merged_df['DATE'] = pd.to_datetime(merged_df['DATE'])

            # Plot forecast
            st.subheader("Real vs Forecasted Sales")
            plot_forecast(merged_df)

            # Model evaluation
            st.subheader("Model Performance Metrics")
            metrics = evaluate_model(best_model, test_data)
            st.write(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%")
            st.write(f"Mean Absolute Error (MAE): {metrics['mae']:.2f}")
            st.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.2f}")

            if metrics['mape'] > 5:
                st.warning("MAPE is above 5%. Consider adding more features or tuning the model further.")
            else:
                st.success("MAPE is below 5%. The model is performing well.")

            # Display model components
            #st.subheader("Model Components")
            #fig = best_model.plot_components(forecast)
            #st.write(fig)

            # Residuals Analysis
            #st.subheader("Residuals Analysis")
            #residuals = metrics['results']['y'] - metrics['results']['yhat']
            #fig_residuals = px.histogram(residuals, title='Residuals Histogram')
            #st.write(fig_residuals)

        else:
            st.error("Failed to load data.")
    else:
        st.info("Please upload a CSV file to proceed.")


if __name__ == "__main__":
    main()