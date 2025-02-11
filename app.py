import streamlit as st
import pandas as pd
import numpy as np
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

# Load API keys from environment variables
load_dotenv()
fred_api_key = os.getenv("FRED_API_KEY")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Streamlit UI
st.title("📈 Portfolio Optimization App")
st.write("This app allows you to construct an optimized portfolio using bonds, options, futures, and swaps.")

# User Input: Select Assets
st.sidebar.header("Select Assets for Portfolio")
data_fetcher = DataFetcher(fred_api_key, alpha_vantage_api_key)

# **Fetch US Treasury Swap Rates**
st.sidebar.subheader("Select US Treasury Swap")
treasury_swaps_df = data_fetcher.fetch_treasury_swaps()
selected_swap = st.sidebar.selectbox(
    "Choose a Treasury Swap", 
    treasury_swaps_df["security_desc"].unique() if not treasury_swaps_df.empty else ["US Treasury Bonds"]
)

# Fetch Data Button
if st.sidebar.button("Fetch Data & Optimize"):
    logger.info("Fetching data...")

    # Fetch and Process Data
    bond_data = data_fetcher.fetch_bond_yields()
    options_data = data_fetcher.fetch_options_data("IBM")
    futures_data = data_fetcher.fetch_futures_data("IBM")

    # **Fetch US Treasury swaps (ensure valid data)**
    if treasury_swaps_df is not None and not treasury_swaps_df.empty:
        swap_data = treasury_swaps_df[treasury_swaps_df["security_desc"] == selected_swap]
    else:
        swap_data = pd.DataFrame()

    # **Ensure swap_data is valid**
    if swap_data.empty:
        st.warning("No Treasury swap data available. Using default values.")
        swap_data = pd.DataFrame({"security_desc": ["US Treasury Bonds"], "avg_interest_rate_amt": [0.02]})

    # Process Data
    processed_bond_data = process_bond_data(bond_data)
    processed_derivative_data = process_derivative_data({
        "options": options_data,
        "futures": futures_data,
        "swaps": swap_data
    })

    # Initialize Pricing Models
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

    # ✅ Fix: Convert swap_prices to float if dictionary
    if isinstance(swap_prices, dict):
        swap_prices = list(swap_prices.values())[0] if swap_prices else 0.0

    # ✅ Ensure all values are float
    option_price = option_prices["Black-Scholes Call"]
    asset_returns = np.array([bond_prices, option_price, futures_prices, swap_prices])

    # ✅ Compute covariance matrix safely
    if len(asset_returns) > 1:
        cov_matrix = np.cov(asset_returns)
    else:
        cov_matrix = np.array([[np.var(asset_returns)]])

    # Perform Portfolio Optimization
    portfolio_optimizer = PortfolioOptimization(asset_returns, cov_matrix)
    optimized_results, optimal_weights = portfolio_optimizer.optimize_portfolio()

    # Display Portfolio Optimization Results
    st.subheader("📊 Optimized Portfolio Weights")
    asset_labels = ["Bonds", "Options", "Futures", "Swaps"]
    optimized_weights = optimized_results[0]

    # ✅ Fix: Ensure correct weight array size
    optimized_weights = np.array(optimized_weights).flatten()
    if len(optimized_weights) != len(asset_labels):
        optimized_weights = optimized_weights[:len(asset_labels)]

    # **Plot Metrics**
    plot_metrics(optimized_weights, asset_labels)
