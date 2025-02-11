import os
import numpy as np
import pandas as pd
import logging
import streamlit as st
from dotenv import load_dotenv

# Import project modules
from data.data_fetching import DataFetcher
from data.data_processing import process_bond_data, process_derivative_data
from models.bond_pricing import BondPricing
from models.option_pricing import OptionPricing
from models.futures_pricing import FuturesPricing
from models.swap_pricing import SwapPricing
from optimization.portfolio_optimization import PortfolioOptimization
from visualization.visual_analysis import plot_metrics

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load API keys from .env file
load_dotenv()
fred_api_key = os.getenv("FRED_API_KEY")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
swap_api_key = os.getenv("SWAP_API_KEY")

# Streamlit UI
st.title("📊 Portfolio Optimization App")
st.write("Use this app to fetch financial data and optimize your portfolio.")

# Button to fetch data
if st.button("📡 Fetch Data & Optimize"):
    st.write("🔄 Fetching financial data...")

    data_fetcher = DataFetcher(fred_api_key, alpha_vantage_api_key, swap_api_key)

    try:
        bond_data = data_fetcher.fetch_bond_yields()
        options_data = data_fetcher.fetch_options_data("IBM")
        futures_data = data_fetcher.fetch_futures_data("IBM")
        swap_data = data_fetcher.fetch_swap_rates()

        st.write("✅ Data fetched successfully!")
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        logger.error(f"Error fetching data: {e}")
        st.stop()

    # Ensure swap_data is valid
    if swap_data is None or swap_data.empty:
        swap_data = pd.DataFrame()
        logger.warning("Swap rates data is empty.")

    # Process Data
    st.write("🔄 Processing data...")
    processed_bond_data = process_bond_data(bond_data)
    processed_derivative_data = process_derivative_data({
        "options": options_data,
        "futures": futures_data,
        "swaps": swap_data
    })
    st.write("✅ Data processing complete!")

    # Initialize Pricing Models with default values
    st.write("🔄 Initializing pricing models...")
    try:
        bond_pricing = BondPricing(1000, 0.05, 10, 0.03)
        option_pricing = OptionPricing(100, 100, 1, 0.03, 0.20)
        futures_pricing = FuturesPricing(100, 100, 1, 0.03, 0.20)
        swap_pricing = SwapPricing(swap_data)
        st.write("✅ Pricing models initialized!")
    except Exception as e:
        st.error(f"❌ Error initializing pricing models: {e}")
        logger.error(f"Error initializing pricing models: {e}")
        st.stop()

    # Calculate Prices
    st.write("🔄 Calculating asset prices...")
    try:
        bond_prices = bond_pricing.price()
        option_prices = option_pricing.black_scholes_call()
        futures_prices = futures_pricing.calculate_futures_price()
        swap_prices = swap_pricing.calculate_prices()
        st.write("✅ Price calculations complete!")
    except Exception as e:
        st.error(f"❌ Error calculating prices: {e}")
        logger.error(f"Error calculating prices: {e}")
        st.stop()

    # Ensure swap_prices is a valid float
    if isinstance(swap_prices, dict):
        swap_prices = list(swap_prices.values())[0] if swap_prices else 0.0

    # Prepare data for optimization
    asset_returns = np.array([bond_prices, option_prices, futures_prices, swap_prices])

    if len(asset_returns) > 1:
        cov_matrix = np.cov(asset_returns)
    else:
        cov_matrix = np.array([[np.var(asset_returns)]])

    # Perform Portfolio Optimization
    st.write("🔄 Optimizing portfolio...")
    try:
        portfolio_optimizer = PortfolioOptimization(asset_returns, cov_matrix)
        optimized_results, optimal_weights = portfolio_optimizer.optimize_portfolio()
        st.write("✅ Portfolio optimization complete!")
    except Exception as e:
        st.error(f"❌ Error in portfolio optimization: {e}")
        logger.error(f"Error in portfolio optimization: {e}")
        st.stop()

    # Display Optimized Portfolio
    st.subheader("📈 Optimized Portfolio Weights")
    asset_labels = ["Bonds", "Options", "Futures", "Swaps"]

    # Extract first optimized portfolio weights
    optimized_weights = optimized_results[0] if optimized_results.ndim == 2 else optimized_results

    # Ensure correct length
    optimized_weights = np.array(optimized_weights).flatten()
    if len(optimized_weights) != len(asset_labels):
        optimized_weights = optimized_weights[:len(asset_labels)]

    # Display Weights as Table
    df_weights = pd.DataFrame({"Asset": asset_labels, "Weight": optimized_weights})
    st.dataframe(df_weights)

    # Display Weights as Bar Chart
    st.bar_chart(df_weights.set_index("Asset"))

    # Plot Metrics
    plot_metrics(optimized_weights, asset_labels)
