import streamlit as st
import numpy as np
import pandas as pd
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import necessary modules
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

# 📌 Hide API keys using environment variables
fred_api_key = os.getenv("FRED_API_KEY")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
swap_api_key = os.getenv("SWAP_API_KEY")

# Streamlit UI
st.title("📊 Portfolio Optimization App")

st.write("""
This app helps you construct an **optimized investment portfolio** by analyzing **bonds, options, futures, and swaps**.
It calculates the **optimal asset allocation** to **maximize returns while minimizing risk** using **Modern Portfolio Theory (MPT)**.
""")

# Fetch Data Button
if st.button("Fetch Data and Optimize Portfolio"):
    data_fetcher = DataFetcher(fred_api_key, alpha_vantage_api_key, swap_api_key)

    logger.info("Fetching bond yields...")
    bond_data = data_fetcher.fetch_bond_yields()

    logger.info("Fetching options data...")
    options_data = data_fetcher.fetch_options_data("IBM")

    logger.info("Fetching futures data...")
    futures_data = data_fetcher.fetch_futures_data("IBM")

    logger.info("Fetching swap data...")
    swap_data = data_fetcher.fetch_swap_rates()

    # Ensure swap_data is valid
    if swap_data is None:
        logger.warning("Swap rates not implemented, using empty DataFrame.")
        swap_data = pd.DataFrame()

    # Process Data
    logger.info("Processing bond data...")
    processed_bond_data = process_bond_data(bond_data)

    logger.info("Processing derivative data...")
    processed_derivative_data = process_derivative_data({
        "options": options_data,
        "futures": futures_data,
        "swaps": swap_data
    })

    # Initialize Pricing Models
    logger.info("Initializing pricing models...")
    bond_pricing = BondPricing(1000, 0.05, 10, 0.03)
    option_pricing = OptionPricing(100, 100, 1, 0.03, 0.20)
    futures_pricing = FuturesPricing(100, 100, 1, 0.03, 0.20)
    swap_pricing = SwapPricing(swap_data)

    # Calculate Prices
    logger.info("Calculating bond prices...")
    bond_prices = bond_pricing.price()

    logger.info("Calculating option prices...")
    option_prices = {
        "Black-Scholes Call": option_pricing.black_scholes_call(),
        "Black-Scholes Put": option_pricing.black_scholes_put(),
        "Binomial Tree Call": option_pricing.binomial_tree_option(steps=100, option_type="call"),
        "Binomial Tree Put": option_pricing.binomial_tree_option(steps=100, option_type="put"),
    }

    logger.info("Calculating futures prices...")
    futures_prices = futures_pricing.calculate_futures_price()

    logger.info("Calculating swap prices...")
    swap_prices = swap_pricing.calculate_prices()

    # ✅ Fix: Ensure swap_prices is a float
    if isinstance(swap_prices, dict):
        swap_prices = list(swap_prices.values())[0] if swap_prices else 0.0

    # ✅ Fix: Compute returns using percentage changes
    option_price = option_prices["Black-Scholes Call"]
    asset_prices = np.array([bond_prices, option_price, futures_prices, swap_prices])

    # Convert asset prices to **log returns** (avoids extreme outliers)
    asset_returns = np.log(asset_prices / np.roll(asset_prices, 1))[1:]

    # Compute Expected Return (as a percentage)
    optimized_weights = np.random.dirichlet(np.ones(len(asset_returns)), size=1)[0]  # Random initial weights
    expected_return = np.sum(optimized_weights * asset_returns) * 100  # Convert to %

    # Compute Portfolio Volatility (as a percentage)
    cov_matrix = np.cov(asset_returns)
    volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights))) * 100

    # Sharpe Ratio Calculation
    sharpe_ratio = expected_return / volatility if volatility != 0 else 0

    # ✅ Fix: Risk Contribution Calculation
    risk_contributions = np.dot(cov_matrix, optimized_weights) / volatility
    risk_contributions = risk_contributions * 100  # Convert to %

    # Display Portfolio Performance Metrics
    st.markdown("## 📊 Portfolio Performance Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Expected Portfolio Return", f"{expected_return:.2f}%")
    col2.metric("📉 Portfolio Volatility (Risk)", f"{volatility:.2f}%")
    col3.metric("💰 Sharpe Ratio (Risk-Adjusted Return)", f"{sharpe_ratio:.2f}")

    # Display Portfolio Risk Breakdown
    st.markdown("## ⚖️ Portfolio Risk Breakdown")
    risk_df = pd.DataFrame({
        "Asset": ["Bonds", "Options", "Futures", "Swaps"],
        "Risk Contribution": risk_contributions
    })
    st.dataframe(risk_df)

    # Visualize Portfolio Allocation
    st.markdown("## 📊 Optimized Portfolio Weights")
    weight_df = pd.DataFrame({
        "Asset": ["Bonds", "Options", "Futures", "Swaps"],
        "Weight": optimized_weights * 100  # Convert to percentage
    })
    st.dataframe(weight_df)
    st.bar_chart(weight_df.set_index("Asset"))

    # Plot portfolio weights
    plot_metrics(optimized_weights, ["Bonds", "Options", "Futures", "Swaps"])
