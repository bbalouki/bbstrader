import numpy as np
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

def load_and_prepare_data(df):
    """
    Prepares financial time series data for analysis.
    
    This function takes a pandas DataFrame containing financial data,
    calculates logarithmic returns, and the first difference 
    of these logarithmic returns. It handles missing values 
    by filling them with zeros.
    
    Parameters
    ==========
    :param df (pd.DataFrame): DataFrame containing at least 
    a 'Close' column with closing prices of a financial asset.
    
    Returns:
    - pd.DataFrame: DataFrame with additional 
        columns for logarithmic returns ('log_return') 
        and the first difference of logarithmic returns ('diff_log_return'), 
        with NaN values filled with 0.
    """
    # Load data
    data = df.copy()
    # Calculate logarithmic returns
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    # Differencing if necessary
    data['diff_log_return'] = data['log_return'].diff()
    # Drop NaN values
    data.fillna(0, inplace=True)
    return data

def fit_best_arima(window_data):
    """
    Identifies and fits the best ARIMA model 
    based on the Akaike Information Criterion (AIC).
    
    Iterates through different combinations of p and q 
    parameters (within specified ranges) for the ARIMA model,
    fits them to the provided data, and selects the combination 
    with the lowest AIC value.
    
    Parameters
    ==========
    :param window_data (pd.Series or np.array): 
        Time series data to fit the ARIMA model on.
    
    Returns:
        - ARIMA result object: The fitted ARIMA 
            model with the lowest AIC.
    """
    model = auto_arima(
        window_data, 
        start_p=1, 
        start_q=1, 
        max_p=6, 
        max_q=6, 
        seasonal=False,   
        stepwise=True
    )
    final_order = model.order
    best_arima_model = ARIMA(window_data, order=final_order, missing='drop').fit()
    return best_arima_model
    
def fit_garch(window_data):
    """
    Fits an ARIMA model to the data to get residuals, 
    then fits a GARCH(1,1) model on these residuals.
    
    Utilizes the residuals from the best ARIMA model fit to 
    then model volatility using a GARCH(1,1) model.
    
    Parameters
    ==========
    :param window_data (pd.Series or np.array): 
        Time series data for which to fit the ARIMA and GARCH models.
    
    Returns
        - tuple: A tuple containing the ARIMA result 
            object and the GARCH result object.
    """
    arima_result = fit_best_arima(window_data)
    resid = np.asarray(arima_result.resid)
    resid = resid[~(np.isnan(resid) | np.isinf(resid))]
    garch_model = arch_model(resid, p=1, q=1, rescale=False)
    garch_result = garch_model.fit(disp='off')
    return arima_result, garch_result

def predict_next_return(arima_result, garch_result):
    """
    Predicts the next return value using fitted ARIMA and GARCH models.
    
    Combines the next period forecast from the ARIMA model 
    with the next period volatility forecast from the GARCH model
    to predict the next return value.
    
    Parameters
    ==========
    :param arima_result (ARIMA result object): The fitted ARIMA model result.
    :param garch_result (ARCH result object): The fitted GARCH model result.
    
    Returns
        - float: The predicted next return, adjusted for predicted volatility.
    """
    # Predict next value with ARIMA
    arima_pred = arima_result.forecast(steps=1)
    # Predict next volatility with GARCH
    garch_pred = garch_result.forecast(horizon=1)
    next_volatility = garch_pred.variance.iloc[-1, 0]

    # Combine predictions (return + volatility)
    next_return = arima_pred.values[0] + next_volatility
    return next_return

def get_prediction(window_data):
    """
    Orchestrator function to get the next period's return prediction.
    
    This function ties together the process of fitting both ARIMA and GARCH models on the provided data
    and then predicting the next period's return using these models.
    
    Parameters
    ==========
    :param window_data (pd.Series or np.array): 
        Time series data to fit the models and predict the next return.
    
    Returns
        - float: Predicted next return value.
    """
    arima_result, garch_result = fit_garch(window_data)
    prediction = predict_next_return(arima_result, garch_result)
    return prediction
