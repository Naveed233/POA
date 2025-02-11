import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from data.data_fetching import DataFetcher
from data.data_processing import process_bond_data, process_derivative_data
from models.bond_pricing import BondPricing
from models.option_pricing import OptionPricing
from models.futures_pricing import FuturesPricing
from models.swap_pricing import SwapPricing
from optimization.portfolio_optimization import PortfolioOptimization
from visualization.visual_analysis import plot_metrics

# Load API keys from .env file
load_dotenv()
fred_api_key = os.getenv("FRED_API_KEY")
alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Streamlit UI
st.title("📊 Portfolio Optimization App")
st.write("Build an optimized portfolio using **Bonds, Options, Futures, and Swaps**. Select assets below and calculate optimal allocations.")

# Initialize DataFetcher
data_fetcher = DataFetcher(fred_api_key, alpha_vantage_api_key)

# Dropdown Selectors for Each Asset Class
st.subheader("📌 Select Assets for Portfolio")

# Fetch Treasury Swaps Data
treasury_swaps_df = data_fetcher.fetch_treasury_swaps()

# Define Dropdown Options
bond_choices = ["US Treasury Bond", "Corporate Bond", "Municipal Bond"]
option_choices = ["IBM Options", "AAPL Options", "GOOGL Options"]
futures_choices = ["S&P 500 Futures", "Gold Futures", "Oil Futures"]
swap_choices = treasury_swaps_df["security_desc"].unique().tolist() if not treasury_swaps_df.empty else ["US Treasury Bonds"]

# User Selections
selected_bond = st.selectbox("📉 Choose a Bond", bond_choices)
selected_option = st.selectbox("📈 Choose an Option", option_choices)
selected_future = st.selectbox("🛢️ Choose a Future", futures_choices)
selected_swap = st.selectbox("🏦 Choose a Swap", swap_choices)

# Fetch Data & Optimize Button
if st.button("🔍 Fetch Data & Optimize"):
    logger.info("Fetching data...")

    # Fetch Market Data
    bond_data = data_fetcher.fetch_bond_yields()
    options_data = data_fetcher.fetch_options_data(selected_option.split(" ")[0])
    futures_data = data_fetcher.fetch_futures_data(selected_future.split(" ")[0])

    # Fetch Swap Rates
    swap_data = treasury_swaps_df[treasury_swaps_df["security_desc"] == selected_swap] if not treasury_swaps_df.empty else pd.DataFrame()

    # Ensure swap_data is valid
    if swap_data.empty:
        st.warning("⚠️ No Treasury swap data available. Using default values.")
        swap_data = pd.DataFrame({"security_desc": ["US Treasury Bonds"], "avg_interest_rate_amt": [0.02]})
    
    if "avg_interest_rate_amt" in swap_data.columns:
        swap_data = swap_data.rename(columns={"avg_interest_rate_amt": "rate"})

    # Update Asset Labels
    bond_label = f"Bonds ({selected_bond})"
    option_label = f"Options ({selected_option})"
    futures_label = f"Futures ({selected_future})"
    swap_label = f"Swaps ({selected_swap})"

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
    option_prices = option_pricing.black_scholes_call()
    futures_prices = futures_pricing.calculate_futures_price()
    swap_prices = swap_pricing.calculate_prices()

    # Ensure correct scaling
    option_price = option_prices
    asset_returns = np.array([bond_prices, option_price, futures_prices, swap_prices]) / 100  # Convert to % scale

    # Compute Covariance Matrix
    cov_matrix = np.cov(asset_returns) if len(asset_returns) > 1 else np.array([[np.var(asset_returns)]])

    # Portfolio Optimization
    portfolio_optimizer = PortfolioOptimization(asset_returns, cov_matrix)
    optimized_results, optimal_weights = portfolio_optimizer.optimize_portfolio()

    # Extract Weights
    asset_labels = [bond_label, option_label, futures_label, swap_label]
    optimized_weights = np.array(optimized_results[0]).flatten()

    # Ensure correct shape
    if len(optimized_weights) != len(asset_labels):
        optimized_weights = optimized_weights[:len(asset_labels)]

    # Display Optimized Portfolio Weights
    st.subheader("📊 Optimized Portfolio Weights")
    plot_metrics(optimized_weights, asset_labels)

    # Portfolio Performance Metrics
    expected_return = np.dot(optimized_weights, asset_returns) * 100  # Convert to %
    portfolio_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights))) * 100  # Convert to %
    sharpe_ratio = expected_return / portfolio_volatility if portfolio_volatility != 0 else 0

    # Display Metrics
    st.subheader("📊 Portfolio Performance Metrics")
    st.metric("📈 Expected Portfolio Return", f"{expected_return:.2f}%")
    st.metric("📉 Portfolio Volatility (Risk)", f"{portfolio_volatility:.2f}%")
    st.metric("💰 Sharpe Ratio (Risk-Adjusted Return)", f"{sharpe_ratio:.2f}")

    # Portfolio Risk Breakdown
    risk_contributions = optimized_weights * portfolio_volatility

    # Display Risk Breakdown
    st.subheader("⚖️ Portfolio Risk Breakdown")
    risk_df = pd.DataFrame({"Asset": asset_labels, "Risk Contribution": risk_contributions})
    st.dataframe(risk_df)

    # Correlation Matrix
    st.subheader("📊 Asset Correlation Heatmap")
    corr_matrix = np.corrcoef(asset_returns) if asset_returns.ndim == 2 else np.array([[1]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, xticklabels=asset_labels, yticklabels=asset_labels, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Efficient Frontier Plot
    st.subheader("📈 Efficient Frontier")
    fig, ax = plt.subplots(figsize=(8, 6))
    risk_range = np.linspace(0, np.max(portfolio_volatility) * 1.5, 100)
    efficient_returns = np.sqrt(risk_range) * sharpe_ratio  # Simulated efficient frontier
    
    ax.plot(risk_range, efficient_returns, label="Efficient Frontier", color="blue")
    ax.scatter(portfolio_volatility, expected_return, marker="o", color="red", label="Optimized Portfolio")
    ax.set_xlabel("Portfolio Volatility (Risk)")
    ax.set_ylabel("Expected Return")
    ax.legend()
    st.pyplot(fig)
