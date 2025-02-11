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
    "This app allows users to **customize** their portfolio by selecting financial assets "
    "such as bonds, options, futures, and swaps (using US Treasury rates). The app fetches "
    "real market data, calculates pricing, and applies **modern portfolio optimization** "
    "techniques to suggest the best asset allocation."
)

# Sidebar for API key inputs (Hidden using environment variables)
fred_api_key = os.getenv("FRED_API_KEY", "")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# Initialize DataFetcher
data_fetcher = DataFetcher(fred_api_key, alpha_vantage_api_key)

# Step 1: Fetch Available Assets
st.sidebar.header("📌 Choose Your Assets")

# Fetch bonds
bonds_list = data_fetcher.fetch_bond_yields().get("security_desc", ["US Treasury 10Y"])
selected_bond = st.sidebar.selectbox("Select a Bond", bonds_list)

# Fetch options
options_list = ["IBM", "AAPL", "TSLA", "AMZN", "GOOGL"]  # Replace with API data if available
selected_option = st.sidebar.selectbox("Select an Option", options_list)

# Fetch futures
futures_list = ["IBM", "AAPL", "S&P 500", "Gold", "Crude Oil"]  # Replace with API data if available
selected_future = st.sidebar.selectbox("Select a Future", futures_list)

# Fetch US Treasury swaps
treasury_swaps_df = data_fetcher.fetch_treasury_swaps()
swaps_list = treasury_swaps_df["security_desc"].unique().tolist() if not treasury_swaps_df.empty else ["US Treasury Bonds"]
selected_swap = st.sidebar.selectbox("Select a Swap (US Treasury)", swaps_list)

# Step 2: Fetch Data & Optimize
if st.button("Fetch Data & Optimize"):
    st.write("📡 Fetching financial data...")

    # Fetch Data
    bond_data = data_fetcher.fetch_bond_yields()
    options_data = data_fetcher.fetch_options_data(selected_option)
    futures_data = data_fetcher.fetch_futures_data(selected_future)

    # Fetch US Treasury swaps
    swap_data = treasury_swaps_df[treasury_swaps_df["security_desc"] == selected_swap] if not treasury_swaps_df.empty else pd.DataFrame()

    # Ensure swap_data is not None
    if swap_data.empty:
        st.warning("No Treasury swap data available. Using default values.")
        swap_data = pd.DataFrame({"security_desc": ["US Treasury Bonds"], "avg_interest_rate_amt": [0.02]})

    # Process Data
    processed_bond_data = process_bond_data(bond_data)
    processed_derivative_data = process_derivative_data({"options": options_data, "futures": futures_data, "swaps": swap_data})

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

    # Fix NoneType issue
    if isinstance(swap_prices, dict):
        swap_prices = list(swap_prices.values())[0] if swap_prices else 0.0

    # Create portfolio
    asset_returns = np.array([bond_prices, option_prices["Black-Scholes Call"], futures_prices, swap_prices])
    cov_matrix = np.cov(asset_returns) if len(asset_returns) > 1 else np.array([[np.var(asset_returns)]])

    # Optimize Portfolio
    portfolio_optimizer = PortfolioOptimization(asset_returns, cov_matrix)
    optimized_results, optimal_weights = portfolio_optimizer.optimize_portfolio()

    # Display Portfolio Metrics
    asset_labels = ["Bonds", "Options", "Futures", selected_swap]
    optimized_weights = np.array(optimized_results[0]).flatten()

    # Fix shape mismatch
    optimized_weights = optimized_weights[:len(asset_labels)]

    # Display the graph
    plot_metrics(optimized_weights, asset_labels)
