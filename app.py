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
    "such as bonds, options, futures, and swaps. The app fetches financial data, calculates pricing, "
    "and applies **modern portfolio optimization** techniques to suggest the best asset allocation."
)

# Sidebar for API key inputs (Hidden using environment variables)
fred_api_key = os.getenv("FRED_API_KEY", "")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
swap_api_key = os.getenv("SWAP_API_KEY", "")

# Step 1: Allow Users to Choose Their Assets
st.sidebar.header("📌 Choose Your Assets")

selected_assets = st.sidebar.multiselect(
    "Select Asset Types",
    options=["Bonds", "Options", "Futures", "Swaps"],
    default=["Bonds", "Options", "Futures"]
)

# Step 2: Allow Users to Enter Specific Asset Symbols
selected_bonds = st.sidebar.text_area("Enter Bond Names (comma-separated)", "US Treasury 10Y")
selected_options = st.sidebar.text_area("Enter Option Symbols (comma-separated)", "IBM")
selected_futures = st.sidebar.text_area("Enter Futures Symbols (comma-separated)", "IBM")
selected_swaps = st.sidebar.text_area("Enter Swap Names (comma-separated)", "LIBOR Swap")

# Convert text input into lists
bonds_list = [x.strip() for x in selected_bonds.split(",") if x.strip()]
options_list = [x.strip() for x in selected_options.split(",") if x.strip()]
futures_list = [x.strip() for x in selected_futures.split(",") if x.strip()]
swaps_list = [x.strip() for x in selected_swaps.split(",") if x.strip()]

# Fetch Data Button
if st.button("Fetch Data & Optimize"):
    st.write("📡 Fetching financial data...")

    data_fetcher = DataFetcher(fred_api_key, alpha_vantage_api_key, swap_api_key)

    bond_data, options_data, futures_data, swap_data = None, None, None, None

    if "Bonds" in selected_assets:
        bond_data = data_fetcher.fetch_bond_yields()

    if "Options" in selected_assets:
        options_data = data_fetcher.fetch_options_data(options_list[0] if options_list else "IBM")

    if "Futures" in selected_assets:
        futures_data = data_fetcher.fetch_futures_data(futures_list[0] if futures_list else "IBM")

    if "Swaps" in selected_assets:
        swap_data = data_fetcher.fetch_swap_rates()

    # Process Data
    processed_bond_data = process_bond_data(bond_data) if bond_data is not None else pd.DataFrame()
    processed_derivative_data = process_derivative_data(
        {"options": options_data, "futures": futures_data, "swaps": swap_data}
    )

    # Extract Asset Labels
    asset_labels = []
    if "Bonds" in selected_assets:
        asset_labels.append(bonds_list[0] if bonds_list else "Generic Bond")
    if "Options" in selected_assets:
        asset_labels.append(options_list[0] if options_list else "IBM Option")
    if "Futures" in selected_assets:
        asset_labels.append(futures_list[0] if futures_list else "IBM Futures")
    if "Swaps" in selected_assets:
        asset_labels.append(swaps_list[0] if swaps_list else "Swap Contract")

    # Initialize Pricing Models
    st.write("🧮 Calculating prices...")

    bond_prices = BondPricing(1000, 0.05, 10, 0.03).price() if "Bonds" in selected_assets else 0
    option_prices = OptionPricing(100, 100, 1, 0.03, 0.20).black_scholes_call() if "Options" in selected_assets else 0
    futures_prices = FuturesPricing(100, 100, 1, 0.03, 0.20).calculate_futures_price() if "Futures" in selected_assets else 0
    swap_prices = SwapPricing(swap_data).calculate_prices() if "Swaps" in selected_assets else 0

    # Convert swap_prices to a usable float value
    if isinstance(swap_prices, dict):
        swap_prices = list(swap_prices.values())[0] if swap_prices else 0.0

    # Convert asset prices to NumPy array
    asset_returns = np.array([val for val in [bond_prices, option_prices, futures_prices, swap_prices] if val != 0])

    # Fix covariance matrix computation
    if len(asset_returns) > 1:
        cov_matrix = np.cov(asset_returns)
    else:
        cov_matrix = np.array([[np.var(asset_returns)]]) if len(asset_returns) > 0 else np.array([[0]])

    # Portfolio Optimization
    st.write("📊 Optimizing Portfolio...")

    if len(asset_returns) > 1:
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

    else:
        st.warning("⚠️ Please select at least **two** assets for optimization.")
