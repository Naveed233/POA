import streamlit as st
import pandas as pd
import numpy as np
import logging
from data.data_fetching import DataFetcher
from data.data_processing import process_bond_data, process_derivative_data
from models.bond_pricing import BondPricing
from models.option_pricing import OptionPricing
from models.futures_pricing import FuturesPricing
from models.swap_pricing import SwapPricing
from optimization.portfolio_optimization import PortfolioOptimization
from visualization.visual_analysis import plot_metrics

# Streamlit UI
st.title("📈 Portfolio Optimization App")
st.subheader("Analyze and Optimize Your Investment Portfolio")

# API Inputs
st.sidebar.header("🔑 API Keys")
fred_api_key = st.sidebar.text_input("FRED API Key", "d636c702a0f9e3e97f55065da983a21c")
alpha_vantage_api_key = st.sidebar.text_input("AlphaVantage API Key", "FXSWUKPJEOUK5C60")
swap_api_key = st.sidebar.text_input("Swap API Key", "2269891a-0d3f-4d02-bb13-8d241f90a142")

# Fetch Data Button
if st.button("🚀 Fetch & Optimize Portfolio"):
    # Initialize Data Fetcher
    data_fetcher = DataFetcher(fred_api_key, alpha_vantage_api_key, swap_api_key)

    # Fetch Data
    st.write("📊 Fetching market data...")
    bond_data = data_fetcher.fetch_bond_yields()
    options_data = data_fetcher.fetch_options_data("IBM")
    futures_data = data_fetcher.fetch_futures_data("IBM")
    swap_data = data_fetcher.fetch_swap_rates()

    # Process Data
    st.write("🔄 Processing market data...")
    processed_bond_data = process_bond_data(bond_data)
    processed_derivative_data = process_derivative_data({
        "options": options_data,
        "futures": futures_data,
        "swaps": swap_data
    })

    # Initialize Pricing Models
    st.write("🧮 Calculating prices...")
    bond_pricing = BondPricing(processed_bond_data)
    option_pricing = OptionPricing(processed_derivative_data['options'])
    futures_pricing = FuturesPricing(processed_derivative_data['futures'])
    swap_pricing = SwapPricing(processed_derivative_data['swaps'])

    # Calculate Prices
    bond_prices = bond_pricing.calculate_prices()
    option_prices = option_pricing.calculate_prices()
    futures_prices = futures_pricing.calculate_futures_price()
    swap_prices = swap_pricing.calculate_prices()

    # Portfolio Optimization
    st.write("📌 Optimizing portfolio...")
    portfolio_optimizer = PortfolioOptimization(bond_prices, option_prices, futures_prices, swap_prices)
    optimized_weights = portfolio_optimizer.optimize()

    # Define Asset Labels
    asset_labels = ["Bonds", "Options", "Futures", "Swaps"]

    # Normalize Weights to 100%
    optimized_weights = np.array(optimized_weights).flatten()
    optimized_weights = optimized_weights / np.sum(optimized_weights) * 100

    # Display Optimized Weights in Table
    st.subheader("✅ Optimized Portfolio Allocation")
    allocation_df = pd.DataFrame({"Asset Type": asset_labels, "Allocation (%)": optimized_weights})
    st.dataframe(allocation_df)

    # Display Portfolio Weights as Bar Chart
    st.subheader("📊 Portfolio Weight Distribution")
    st.bar_chart(pd.DataFrame(optimized_weights, index=asset_labels, columns=["Weight (%)"]))

    # Plot Graph with Proper Labels
    plot_metrics(optimized_weights, asset_labels)
