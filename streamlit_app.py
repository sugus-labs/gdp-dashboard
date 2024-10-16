import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import plotly.express as px
from datetime import timedelta
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.model_selection import ParameterGrid
import scipy.stats as stats
import ephem
import matplotlib.pyplot as plt

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

def calculate_lunar_phase(dates):
    """
    Calculate the lunar phase for each date.
    """
    phases = []
    for date in dates:
        moon = ephem.Moon(date)
        moon_phase = moon.phase  # Returns the phase of the moon (0-29.53 days)
        phases.append(moon_phase)
    return phases

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

    # Apply log transformation to stabilize variance
    df['y'] = np.log(df['y'])

    # Winsorize outliers to cap extreme values at 5th and 95th percentiles
    lower_percentile = df['y'].quantile(0.05)
    upper_percentile = df['y'].quantile(0.95)
    df['y'] = np.clip(df['y'], lower_percentile, upper_percentile)

    # Calculate lunar phases
    df['lunar_phase'] = calculate_lunar_phase(df['ds'])

    # Optionally, apply Z-score outlier detection (optional)
    df = df[(np.abs(stats.zscore(df['y'])) < 3)]  # Remove rows where Z-score is too high (outliers)

    # Add additional regressors (for seasonality and trend factors)
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
    model.add_regressor('lunar_phase')  # Ensure this line is included

    with st.spinner("Training the model..."):
        model.fit(train_data)
    return model


def make_forecast(model, periods, include_history=True):
    """
    Make future predictions using the trained model.
    """
    future = model.make_future_dataframe(periods=periods, include_history=include_history)

    # Ensure future DataFrame includes the necessary regressors
    future['day_of_week'] = future['ds'].dt.dayofweek
    future['month'] = future['ds'].dt.month
    future['lunar_phase'] = calculate_lunar_phase(future['ds'])  # Calculate lunar phase for future dates

    forecast = model.predict(future)
    return forecast

def evaluate_model(model, test_data):
    """
    Evaluate the model using test data.
    """
    # Make predictions
    future_test = model.make_future_dataframe(periods=len(test_data), include_history=False)

    # Ensure future_test DataFrame includes the necessary regressors
    future_test['day_of_week'] = future_test['ds'].dt.dayofweek
    future_test['month'] = future_test['ds'].dt.month
    future_test['lunar_phase'] = calculate_lunar_phase(future_test['ds'])  # Calculate lunar phase for test data

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
        model.add_regressor('lunar_phase')  # Add lunar phase
        model.fit(train_data)

        # Evaluate model
        evaluation = evaluate_model(model, test_data)
        mape = evaluation['mape']

        if mape < best_mape:
            best_mape = mape
            best_params = params
            best_model = model

    st.write(f"Best MAPE after hyperparameter tuning: {best_mape:.2f}%")
    st.json(best_params)
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

    fig.update_traces(line=dict(color='blue'), selector=dict(name='REAL SALES'))
    fig.update_traces(line=dict(color='green'), selector=dict(name='FORECASTED SALES'))

    st.plotly_chart(fig, use_container_width=True)


def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Sales Forecasting for ARMADA")

    simple, clustered = st.tabs(["Simple", "Clustered"])

    with simple:

        # File upload
        #st.subheader("Upload your time series data")
        #uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        uploaded_file = 'data/invoices-headers-v1.csv'  

        if uploaded_file is not None:
            # Load data
            sales_df = load_data(uploaded_file)  # Pass the uploaded file here
            if sales_df is not None:
                date_col = 'DATE'
                target_col = 'AMOUNT'

                # Convert date column to datetime and remove timezone
                sales_df[date_col] = pd.to_datetime(sales_df[date_col]).dt.tz_localize(None)

                # Preprocess data
                prophet_data = preprocess_data(sales_df, date_col, target_col)

                # Ensure enough data is available
                if len(prophet_data) < 30:
                    st.error("Not enough data points. Please provide more data.")
                    return

                # Train-Test Split
                test_size = st.sidebar.slider("Select the number of periods for testing", min_value=1, max_value=365, value=12)

                # Split into train and test
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
                    changepoint_prior = st.sidebar.slider(
                        "Changepoint Prior Scale", min_value=0.001, max_value=0.5, value=0.05, step=0.001
                    )
                    seasonality_prior = st.sidebar.slider(
                        "Seasonality Prior Scale", min_value=0.01, max_value=10.0, value=5.0, step=0.01
                    )
                    n_changepoints = st.sidebar.selectbox(
                        "Number of Changepoints", options=[25, 50, 100], index=1
                    )
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

                #if metrics['mape'] > 5:
                #    st.warning("MAPE is above 5%. Consider adding more features or tuning the model further.")
                #else:
                #    st.success("MAPE is below 5%. The model is performing well.")

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

    with clustered:

        #Define functions to calculate Ramadan and Eid al-Adha dates
        def get_ramadan_start(year):
            o = ephem.Observer()
            o.lat, o.lon = '21.4225', '39.8262'  # Mecca coordinates
            o.horizon = '0'
            ramadan_start = ephem.next_new_moon(f'{year}/03/01')  # Approximate start of Ramadan
            return ramadan_start.datetime()

        def get_eid_al_adha(year):
            o = ephem.Observer()
            o.lat, o.lon = '21.4225', '39.8262'  # Mecca coordinates
            eid_al_adha_start = ephem.next_new_moon(f'{year}/06/01')  # Approximate start of Dhu al-Hijjah
            return eid_al_adha_start.datetime()

        # Define Islamic holidays
        years = [2021, 2022, 2023, 2024]
        ramadan_dates = pd.DataFrame({
            'holiday': 'Ramadan',
            'ds': [get_ramadan_start(year) for year in years],
            'lower_window': 0,
            'upper_window': 34  # 35-day Ramadan period
        })

        eid_al_adha_dates = pd.DataFrame({
            'holiday': 'Eid al-Adha',
            'ds': [get_eid_al_adha(year) for year in years],
            'lower_window': -5,  # 5 days before Eid al-Adha
            'upper_window': 0
        })

        # Combine all Islamic holidays into one DataFrame
        islamic_holidays = pd.concat([ramadan_dates, eid_al_adha_dates])

        # Define public holidays for each country (add more countries as needed)
        bahrain_public_holidays = pd.DataFrame({
            'holiday': ['Bahrain National Day'],
            'ds': pd.to_datetime(['2023-12-16']),
            'lower_window': 0,
            'upper_window': 0
        })

        kuwait_public_holidays = pd.DataFrame({
            'holiday': ['Kuwait National Day', 'Liberation Day'],
            'ds': pd.to_datetime(['2023-02-25', '2023-02-26']),
            'lower_window': 0,
            'upper_window': 0
        })

        oman_public_holidays = pd.DataFrame({
            'holiday': ['Oman National Day'],
            'ds': pd.to_datetime(['2023-11-18']),
            'lower_window': 0,
            'upper_window': 0
        })

        qatar_public_holidays = pd.DataFrame({
            'holiday': ['Qatar National Day'],
            'ds': pd.to_datetime(['2023-12-18']),
            'lower_window': 0,
            'upper_window': 0
        })

        saudi_public_holidays = pd.DataFrame({
            'holiday': ['Saudi National Day'],
            'ds': pd.to_datetime(['2023-09-23']),
            'lower_window': 0,
            'upper_window': 0
        })

        uae_public_holidays = pd.DataFrame({
            'holiday': ['UAE National Day'],
            'ds': pd.to_datetime(['2023-12-02']),
            'lower_window': 0,
            'upper_window': 0
        })

        # Combine Islamic holidays with public holidays for each country
        bahrain_holidays = pd.concat([islamic_holidays, bahrain_public_holidays])
        kuwait_holidays = pd.concat([islamic_holidays, kuwait_public_holidays])
        oman_holidays = pd.concat([islamic_holidays, oman_public_holidays])
        qatar_holidays = pd.concat([islamic_holidays, qatar_public_holidays])
        saudi_holidays = pd.concat([islamic_holidays, saudi_public_holidays])
        uae_holidays = pd.concat([islamic_holidays, uae_public_holidays])

        # Map the holidays to their respective country codes
        country_holidays = {
            'BAH': bahrain_holidays,
            'KWT': kuwait_holidays,
            'OMN': oman_holidays,
            'QA': qatar_holidays,
            'SUD': saudi_holidays,
            'UAE': uae_holidays
        }


        df_sales = pd.read_csv('data/sales.csv') 
        df_sales['INVOICE_DT'] = pd.to_datetime(df_sales['INVOICE_DT'])

        # Melt the data for time series forecasting
        df_melted = df_sales.melt(id_vars=['INVOICE_DT', 'COUNTRY_CODE'], 
                                value_vars=['CHOICE', 'OTHER BRAND', 'RIVA', 'RIVA ACCESSORIES', 'RIVA HOME', 'RIVA KIDS'],
                                var_name='BRAND', 
                                value_name='SALES')

        # Replace NaN sales values with 0
        df_melted['SALES'].fillna(0, inplace=True)

        # Add weekend binary features (Friday, Saturday) to capture weekend sales boost
        df_melted['is_friday'] = (df_melted['INVOICE_DT'].dt.weekday == 4).astype(int)
        df_melted['is_saturday'] = (df_melted['INVOICE_DT'].dt.weekday == 5).astype(int)

        # Sort data by date
        df_melted.sort_values(by='INVOICE_DT', inplace=True)


        def is_eid_al_adha(x, eid_dates):
            # Check if the date falls within 5 days before or after any of the Eid al-Adha dates
            return any((eid_date - pd.Timedelta(days=5) <= x <= eid_date + pd.Timedelta(days=5)) for eid_date in eid_dates)

        # Define forecasting function using Prophet with country-specific holidays and regressors
        def forecast_sales_with_holidays(df, country, brand):
            df_filtered = df[(df['COUNTRY_CODE'] == country) & (df['BRAND'] == brand)]
            # Prepare Prophet DataFrame
            df_prophet = df_filtered[['INVOICE_DT', 'SALES', 'is_friday', 'is_saturday']].rename(columns={'INVOICE_DT': 'ds', 'SALES': 'y'})
            print(df_prophet.head())
            df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)
            df_prophet['floor'] = 0  # Set floor to prevent negative values

            # Use the appropriate holiday DataFrame for the selected country
            holidays = country_holidays.get(country, pd.DataFrame())

            # Initialize Prophet model with holidays
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, holidays=holidays)
            
            # Add custom seasonality for Ramadan using Fourier series
            model.add_seasonality(name='ramadan', period=355, fourier_order=10, prior_scale=10)  # Custom seasonality for Ramadan
            # Add custom seasonality for year-end sales (Dec 20 to Jan 1)
            model.add_seasonality(name='year_end_sales', period=365.25, fourier_order=5, 
                                condition_name='year_end_sales')
            
            # Add custom seasonality for Eid al-Adha
            model.add_seasonality(name='eid_al_adha', period=355, fourier_order=5, 
                                condition_name='eid_al_adha')
            
            model.add_regressor('is_friday')
            model.add_regressor('is_saturday')
            
            # Add a binary column for year-end sales period
            df_prophet['year_end_sales'] = ((df_prophet['ds'].dt.month == 12) & (df_prophet['ds'].dt.day >= 20)) | ((df_prophet['ds'].dt.month == 1) & (df_prophet['ds'].dt.day <= 1)).astype(int)
            
            # Add a binary column for Eid al-Adha seasonality (5 days before and after Eid al-Adha)
            eid_al_adha_dates = pd.to_datetime([get_eid_al_adha(year) for year in df_prophet['ds'].dt.year.unique()])
            # Apply the check to determine if the date is within the Eid al-Adha range
            df_prophet['eid_al_adha'] = df_prophet['ds'].apply(lambda x: is_eid_al_adha(x, eid_al_adha_dates))

            # Fit the model
            model.fit(df_prophet)

            # Create future dataframe for 364 days forecast
            future = model.make_future_dataframe(periods=364)
            future['is_friday'] = (future['ds'].dt.weekday == 4).astype(int)
            future['is_saturday'] = (future['ds'].dt.weekday == 5).astype(int)
            future['year_end_sales'] = ((future['ds'].dt.month == 12) & (future['ds'].dt.day >= 20)) | ((future['ds'].dt.month == 1) & (future['ds'].dt.day <= 1)).astype(int)
            future['eid_al_adha'] = future['ds'].apply(lambda x: is_eid_al_adha(x, eid_al_adha_dates))
            
            # Generate forecast
            forecast = model.predict(future)

            # Clip negative values from forecast
            forecast[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].abs()
            print('forecasted:')
            print(forecast.head())

            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        # Streamlit application
        st.title('Sales Forecasting Dashboard')

        # User selects country and brand
        selected_country = st.selectbox('Select Country', df_melted['COUNTRY_CODE'].unique())
        selected_brand = st.selectbox('Select Brand', df_melted['BRAND'].unique())

        # Generate forecast
        forecast_df = forecast_sales_with_holidays(df_melted, selected_country, selected_brand)

        # Plot actual vs forecasted sales
        st.subheader(f'Actual vs. Forecasted Sales for {selected_brand} in {selected_country}')
        fig, ax = plt.subplots()
        actual_sales = df_melted[(df_melted['COUNTRY_CODE'] == selected_country) & (df_melted['BRAND'] == selected_brand)]

        # Plot actual sales
        ax.plot(actual_sales['INVOICE_DT'], actual_sales['SALES'], label='Actual Sales', color='blue')

        # Plot forecasted sales
        ax.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecasted Sales', color='green')
        ax.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], color='lightgreen', alpha=0.3, label='Confidence Interval')

        # Labels and legend
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.legend()

        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()