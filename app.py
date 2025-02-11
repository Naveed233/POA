import streamlit as st
import numpy as np
import pandas as pd
import logging
import os
from dotenv import load_dotenv
from data.data_fetching import DataFetcher
from data.data_processing import process_bond_data, process_derivative_data
from models.bond_pricing import BondPricing
from models.option_pricing import OptionPricing
from models.futures_pricing import FuturesPricing
from models.swap_pricing import SwapPricing
from optimization.portfolio_optimization import PortfolioOptimization
from visualization.visual_analysis import plot_metrics

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Streamlit UI
st.title("📊 Portfolio Optimization App")
st.write(
    "This app helps users optimize financial portfolios by analyzing assets like "
    "bonds, options, futures, and swaps. It fetches financial data, calculates pricing, "
    "and applies optimization techniques to improve portfolio allocation."
)

# Sidebar for API key inputs (Hidden using environment variables)
fred_api_key = os.getenv("FRED_API_KEY", "")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
swap_api_key = os.getenv("SWAP_API_KEY", "")

# Fetch Data Button
if st.button("Fetch Data & Optimize"):
    st.write("📡 Fetching financial data...")

    data_fetcher = DataFetcher(fred_api_key, alpha_vantage_api_key, swap_api_key)

    bond_data = data_fetcher.fetch_bond_yields()
    options_data = data_fetcher.fetch_options_data("IBM")
    futures_data = data_fetcher.fetch_futures_data("IBM")
    swap_data = data_fetcher.fetch_swap_rates()

    # Process Data
    processed_bond_data = process_bond_data(bond_data)
    processed_derivative_data = process_derivative_data(
        {"options": options_data, "futures": futures_data, "swaps": swap_data}
    )

    # Extract Asset Details
    bond_name = processed_bond_data.iloc[0].get("name", "Generic Bond") if not processed_bond_data.empty else "Generic Bond"
    option_name = "IBM Option"
    futures_name = "IBM Futures"
    swap_name = "Swap Contract"

    asset_labels = [bond_name, option_name, futures_name, swap_name]

    # Initialize Pricing Models
    st.write("🧮 Calculating prices...")

    bond_pricing = BondPricing(1000, 0.05, 10, 0.03)
    option_pricing = OptionPricing(100, 100, 1, 0.03, 0.20)
    futures_pricing = FuturesPricing(100, 100, 1, 0.03, 0.20)
    swap_pricing = SwapPricing(swap_data)

    # Calculate Prices
    bond_prices = bond_pricing.price()
    option_prices = {
        "Black-Scholes Call": option_pricing.black_scholes_call(),
        "Black-Scholes Put": option_pricing.black_scholes_put(),
        "Binomial Tree Call": option_pricing.binomial_tree_option(steps=100, option_type="call"),
        "Binomial Tree Put": option_pricing.binomial_tree_option(steps=100, option_type="put"),
    }
    futures_prices = futures_pricing.calculate_futures_price()
    swap_prices = swap_pricing.calculate_prices()

    # Convert swap_prices to a usable float value
    if isinstance(swap_prices, dict):
        swap_prices = list(swap_prices.values())[0] if swap_prices else 0.0

    # Extract correct option price
    option_price = option_prices["Black-Scholes Call"]

    # Convert asset prices to a NumPy array
    asset_returns = np.array([bond_prices, option_price, futures_prices, swap_prices])

    # Fix covariance matrix computation
    if len(asset_returns) > 1:
        cov_matrix = np.cov(asset_returns)
    else:
        cov_matrix = np.array([[np.var(asset_returns)]])

    # Portfolio Optimization
    st.write("📊 Optimizing Portfolio...")

    portfolio_optimizer = PortfolioOptimization(asset_returns, cov_matrix)
    optimized_results, optimal_weights = portfolio_optimizer.optimize_portfolio()

    # Ensure optimized weights are in the correct format
    optimized_weights = np.array(optimized_results[0]).flatten()
    if len(optimized_weights) != len(asset_labels):
        optimized_weights = optimized_weights[:len(asset_labels)]

    # Display Results
    st.header("📊 Optimized Portfolio Weights")
    df_weights = pd.DataFrame({"Asset": asset_labels, "Weight": optimized_weights})
    st.dataframe(df_weights)

    # Plot Optimized Portfolio
    plot_metrics(optimized_weights, asset_labels)

    # Portfolio Performance Metrics
    st.header("📈 Portfolio Performance Metrics")
    expected_return = np.sum(optimized_weights * asset_returns)
    portfolio_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
    sharpe_ratio = expected_return / portfolio_volatility if portfolio_volatility != 0 else 0

    st.metric("📈 Expected Portfolio Return", f"{expected_return * 100:.2f}%")
    st.metric("📊 Portfolio Volatility (Risk)", f"{portfolio_volatility * 100:.2f}%")
    st.metric("💰 Sharpe Ratio (Risk-Adjusted Return)", f"{sharpe_ratio:.2f}")

    # Risk Contribution Breakdown
    st.header("⚖️ Portfolio Risk Breakdown")
    risk_contribution = optimized_weights * portfolio_volatility
    df_risk = pd.DataFrame({"Asset": asset_labels, "Risk Contribution": risk_contribution})
    st.dataframe(df_risk)
