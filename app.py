import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
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

data_fetcher = DataFetcher(fred_api_key, alpha_vantage_api_key)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# **Streamlit UI**
st.title("📊 Portfolio Optimization App")
st.write("Build an optimized portfolio using **Bonds, Options, Futures, and Swaps.** Select assets below and calculate their optimal allocation.")

# **Fetch available options**
bond_choices = ["US Treasury Bond", "Corporate Bond", "Municipal Bond"]
option_choices = ["IBM Options", "AAPL Options", "GOOGL Options"]
futures_choices = ["S&P 500 Futures", "Gold Futures", "Oil Futures"]
treasury_swaps_df = data_fetcher.fetch_treasury_swaps()

swap_choices = (
    treasury_swaps_df["security_desc"].unique().tolist()
    if treasury_swaps_df is not None and not treasury_swaps_df.empty else ["US Treasury Bonds"]
)

# **User Selection**
st.subheader("📌 Select Assets for Portfolio")
selected_bond = st.selectbox("📉 Choose a Bond", bond_choices)
selected_option = st.selectbox("📈 Choose an Option", option_choices)
selected_future = st.selectbox("🛢️ Choose a Future", futures_choices)
selected_swap = st.selectbox("🏦 Choose a Swap", swap_choices)

# **Fetch Data & Optimize Button**
if st.button("🔍 Fetch Data & Optimize"):
    logger.info("Fetching data...")
    
    bond_data = data_fetcher.fetch_bond_yields()
    options_data = data_fetcher.fetch_options_data(selected_option.split(" ")[0])
    futures_data = data_fetcher.fetch_futures_data(selected_future.split(" ")[0])
    
    swap_data = treasury_swaps_df[treasury_swaps_df["security_desc"] == selected_swap] if treasury_swaps_df is not None and not treasury_swaps_df.empty else pd.DataFrame()
    
    if swap_data.empty:
        st.warning("⚠️ No Treasury swap data available. Using default values.")
        swap_data = pd.DataFrame({"security_desc": ["US Treasury Bonds"], "avg_interest_rate_amt": [0.02]})
    
    if "avg_interest_rate_amt" in swap_data.columns:
        swap_data = swap_data.rename(columns={"avg_interest_rate_amt": "rate"})

    # **Process Data**
    processed_bond_data = process_bond_data(bond_data)
    processed_derivative_data = process_derivative_data({
        "options": options_data,
        "futures": futures_data,
        "swaps": swap_data
    })

    # **Initialize Pricing Models**
    bond_pricing = BondPricing(1000, 0.05, 10, 0.03)
    option_pricing = OptionPricing(100, 100, 1, 0.03, 0.20)
    futures_pricing = FuturesPricing(100, 100, 1, 0.03, 0.20)
    swap_pricing = SwapPricing(swap_data)

    # **Calculate Prices**
    bond_prices = bond_pricing.price()
    option_prices = option_pricing.black_scholes_call()
    futures_prices = futures_pricing.calculate_futures_price()
    swap_prices = swap_pricing.calculate_prices()

    if isinstance(swap_prices, dict):
        swap_prices = list(swap_prices.values())[0] if swap_prices else 0.0

    asset_returns = np.array([bond_prices, option_prices, futures_prices, swap_prices])
    cov_matrix = np.cov(asset_returns) if len(asset_returns) > 1 else np.array([[np.var(asset_returns)]])

    portfolio_optimizer = PortfolioOptimization(asset_returns, cov_matrix)
    optimized_results, optimal_weights = portfolio_optimizer.optimize_portfolio()
    
    asset_labels = [
        f"Bonds ({selected_bond})", 
        f"Options ({selected_option})", 
        f"Futures ({selected_future})", 
        f"Swaps ({selected_swap})"
    ]
    optimized_weights = np.array(optimized_results[0]).flatten()
    optimized_weights = optimized_weights[:len(asset_labels)] if len(optimized_weights) != len(asset_labels) else optimized_weights
    
    # **Display Metrics**
    st.subheader("📊 Portfolio Performance Metrics")
    expected_return = np.mean(asset_returns) * 100
    risk = np.std(asset_returns) * 100
    sharpe_ratio = expected_return / risk if risk != 0 else 0
    
    st.metric(label="📈 Expected Portfolio Return", value=f"{expected_return:.2f}%")
    st.metric(label="📉 Portfolio Volatility (Risk)", value=f"{risk:.2f}%")
    st.metric(label="💰 Sharpe Ratio (Risk-Adjusted Return)", value=f"{sharpe_ratio:.2f}")
    
    # **Visualization: Portfolio Weights**
    st.subheader("📊 Optimized Portfolio Weights")
    plot_metrics(optimized_weights, asset_labels)
    
    # **Visualization: Risk Contribution Pie Chart**
    st.subheader("⚖️ Portfolio Risk Breakdown")
    risk_contributions = (optimized_weights * risk) / np.sum(optimized_weights * risk)
    risk_df = pd.DataFrame({"Asset": asset_labels, "Risk Contribution": risk_contributions})
    st.dataframe(risk_df)
    
    fig, ax = plt.subplots()
    ax.pie(risk_contributions, labels=asset_labels, autopct="%1.1f%%", startangle=140)
    ax.axis("equal")
    st.pyplot(fig)
    
    # **Visualization: Correlation Heatmap**
    st.subheader("🔗 Correlation Matrix")
    corr_matrix = np.corrcoef(asset_returns)
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, xticklabels=asset_labels, yticklabels=asset_labels, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    # **Visualization: Efficient Frontier**
    st.subheader("📈 Efficient Frontier")
    portfolio_optimizer.plot_efficient_frontier()
    
    st.success("✅ Portfolio Optimization Completed!")
