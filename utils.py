import pandas as pd
import numpy as np
import yfinance as yf
import json
import os # Make sure os is imported

# Define quantitative parameters for different risk profiles (as discussed)
RISK_PROFILE_MAPPING = {
    'Conservative': {
        'max_portfolio_vol': 0.15,      # Max annual portfolio volatility (e.g., 15%)
        'max_drawdown_alert': 0.10,     # Max drawdown tolerance (e.g., 10%)
        'max_position_size': 0.08,      # Tighter max individual position (e.g., 8%)
        'min_position_size': 0.005,     # Min individual position (can remain low)
        'min_active_positions': 10,     # Encourage more diversification (e.g., at least 10 stocks)
        'optimization_preference_bias': 'min_variance' # Bias towards safety in method selection
    },
    'Moderate': {
        'max_portfolio_vol': 0.25,      # Moderate annual portfolio volatility (e.g., 25%)
        'max_drawdown_alert': 0.20,     # Moderate drawdown tolerance (e.g., 20%)
        'max_position_size': 0.12,      # Moderate max individual position (e.g., 12%)
        'min_position_size': 0.005,
        'min_active_positions': 5,      # Allow reasonable diversification
        'optimization_preference_bias': 'sharpe_ratio' # Bias towards balanced risk-return (conceptually)
    },
    'Aggressive': {
        'max_portfolio_vol': 0.40,      # Higher annual portfolio volatility (e.g., 40%)
        'max_drawdown_alert': 0.35,     # Higher drawdown tolerance (e.g., 35%)
        'max_position_size': 0.20,      # Higher max individual position (e.g., 20%)
        'min_position_size': 0.005,
        'min_active_positions': 3,      # Allow fewer, more concentrated positions
        'optimization_preference_bias': 'momentum_weighted' # Bias towards growth/higher returns
    }
}


def convert_weights_to_shares(portfolio_weights_series: pd.Series, total_investment_amount: float) -> pd.DataFrame:
    """
    Converts optimized portfolio weights (percentages) into actual share quantities
    based on current market prices.

    Args:
        portfolio_weights_series (pd.Series): A pandas Series where the index contains
                                       stock tickers (e.g., 'TCS.NS') and values are
                                       their optimal weights (as decimals, e.g., 0.05 for 5%).
        total_investment_amount (float): The total amount of money the user
                                         intends to invest (e.g., 100000.0 for ₹1,00,000).

    Returns:
        pd.DataFrame: A DataFrame containing 'stock_symbol', 'shares_to_buy' (integer),
                      'invested_amount' (actual amount spent on that stock), and
                      'final_weight_pct' (actual percentage in the portfolio after rounding).
                      Returns an empty DataFrame if no valid stocks can be processed.
    """
    if portfolio_weights_series.empty or total_investment_amount <= 0:
        return pd.DataFrame(columns=['stock_symbol', 'shares_to_buy', 'invested_amount', 'final_weight_pct'])

    tickers_list = portfolio_weights_series.index.tolist()
    current_prices = {}
    valid_tickers_weights = {}

    # Fetch prices in batches to be more efficient and respect API limits
    batch_size = 50
    for i in range(0, len(tickers_list), batch_size):
        batch_symbols = tickers_list[i:i+batch_size]
        for ticker_symbol in batch_symbols:
            try:
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info
                price = info.get('currentPrice') or info.get('previousClose')
                
                if price is not None and price > 0:
                    current_prices[ticker_symbol] = price
                    valid_tickers_weights[ticker_symbol] = portfolio_weights_series[ticker_symbol]
                else:
                    # st.warning(f"  Warning: Skipping {ticker_symbol} - No valid current price found (possibly delisted or API issue).")
                    pass # Suppress warning for Streamlit app
            except Exception as e:
                # st.error(f"  Error fetching price for {ticker_symbol}: {e}")
                pass # Suppress error for Streamlit app
        if len(tickers_list) > batch_size * 2:
            import time
            time.sleep(0.5)

    if not valid_tickers_weights:
        return pd.DataFrame(columns=['stock_symbol', 'shares_to_buy', 'invested_amount', 'final_weight_pct'])

    valid_weights_series = pd.Series(valid_tickers_weights)
    valid_prices_series = pd.Series(current_prices)

    initial_sum_valid_weights = valid_weights_series.sum()
    if initial_sum_valid_weights == 0:
        return pd.DataFrame(columns=['stock_symbol', 'shares_to_buy', 'invested_amount', 'final_weight_pct'])
        
    normalized_weights = valid_weights_series / initial_sum_valid_weights
    target_monetary_amounts = normalized_weights * total_investment_amount
    raw_shares = target_monetary_amounts / valid_prices_series
    shares_to_buy = np.floor(raw_shares)
    actual_invested_amounts_per_stock = shares_to_buy * valid_prices_series
    total_actual_invested_amount = actual_invested_amounts_per_stock.sum()

    if total_actual_invested_amount == 0:
        final_weights_pct = pd.Series(0.0, index=shares_to_buy.index)
    else:
        final_weights_pct = (actual_invested_amounts_per_stock / total_actual_invested_amount) * 100

    results_df = pd.DataFrame({
        'stock_symbol': shares_to_buy.index,
        'shares_to_buy': shares_to_buy.astype(int),
        'invested_amount': actual_invested_amounts_per_stock,
        'final_weight_pct': final_weights_pct
    })
    
    results_df = results_df[results_df['shares_to_buy'] > 0].reset_index(drop=True)
    
    return results_df
