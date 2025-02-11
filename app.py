import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
from dotenv import load_dotenv  # ✅ Import dotenv
from data.data_fetching import DataFetcher
from data.data_processing import process_bond_data, process_derivative_data
from models.bond_pricing import BondPricing
from models.option_pricing import OptionPricing
from models.futures_pricing import FuturesPricing
from models.swap_pricing import SwapPricing
from optimization.portfolio_optimization import PortfolioOptimization
from visualization.visual_analysis import plot_metrics

# ✅ Load API keys from .env (if available)
load_dotenv()

# Streamlit UI
st.title("📊 Portfolio Optimization App")

# Sidebar for API keys (override if needed)
st.sidebar.header("🔑 API Key Configuration")

fred_api_key = st.sidebar.text_input("FRED API Key", os.getenv("FRED_API_KEY", ""))
alpha_vantage_api_key = st.sidebar.text_input("AlphaVantage API Key", os.getenv("ALPHA_VANTAGE_API_KEY", ""))
swap_api_key = st.sidebar.text_input("Swap API Key", os.getenv("SWAP_API_KEY", ""))

# ✅ Ensure API keys are provided
if not fred_api_key or not alpha_vantage_api_key or not swap_api_key:
    st.error("⚠️ Please enter all required API keys in the sidebar.")

else:
    if st.button("🚀 Fetch Data & Optimize Portfolio"):
        data_fetcher = DataFetcher(fred_api_key, alpha_vantage_api_key, swap_api_key)
        
        st.write("📥 Fetching data...")
        bond_data = data_fetcher.fetch_bond_yields()
        options_data = data_fetcher.fetch_options_data("IBM")
        futures_data = data_fetcher.fetch_futures_data("IBM")
        swap_data = data_fetcher.fetch_swap_rates()

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
        st.write("📈 Computing portfolio weights...")
        bond_prices = bond_pricing.price()
        option_prices = {
            "Black-Scholes Call": option_pricing.black_scholes_call(),
            "Black-Scholes Put": option_pricing.black_scholes_put(),
            "Binomial Tree Call": option_pricing.binomial_tree_option(steps=100, option_type="call"),
            "Binomial Tree Put": option_pricing.binomial_tree_option(steps=100, option_type="put"),
        }
        futures_prices = futures_pricing.calculate_futures_price()
        swap_prices = swap_pricing.calculate_prices()

        # Convert swap prices to float if needed
        if isinstance(swap_prices, dict):
            swap_prices = list(swap_prices.values())[0] if swap_prices else 0.0

        # Compute asset returns
        option_price = option_prices["Black-Scholes Call"]
        asset_returns = np.array([bond_prices, option_price, futures_prices, swap_prices])

        # Compute covariance matrix safely
        cov_matrix = np.cov(asset_returns) if len(asset_returns) > 1 else np.array([[np.var(asset_returns)]])

        # Perform Portfolio Optimization
        portfolio_optimizer = PortfolioOptimization(asset_returns, cov_matrix)
        optimized_results, optimal_weights = portfolio_optimizer.optimize_portfolio()

        # Visualize Portfolio Allocation
        asset_labels = ["Bonds", "Options", "Futures", "Swaps"]
        optimized_weights = optimized_results[0]  # Assume first row contains best weights
        optimized_weights = np.array(optimized_weights).flatten()

        if len(optimized_weights) != len(asset_labels):
            optimized_weights = optimized_weights[:len(asset_labels)]

        plot_metrics(optimized_weights, asset_labels)
