import numpy as np
import pandas as pd
import logging
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
# Ensure swap_data is valid
    if swap_data is None:
        logger.warning("Swap rates not implemented, using empty DataFrame.")
        swap_data = pd.DataFrame()
        
def main():
    # API Keys
    fred_api_key = "d636c702a0f9e3e97f55065da983a21c"
    alpha_vantage_api_key = "FXSWUKPJEOUK5C60"
    swap_api_key = "2269891a-0d3f-4d02-bb13-8d241f90a142"

    data_fetcher = DataFetcher(fred_api_key, alpha_vantage_api_key, swap_api_key)

    # Fetch Data
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

    # ✅ Fix: Use float values for asset returns
    option_price = option_prices["Black-Scholes Call"]
    asset_returns = np.array([bond_prices, option_price, futures_prices, swap_prices])

    # ✅ Fix: Compute covariance matrix safely
    if len(asset_returns) > 1:
        cov_matrix = np.cov(asset_returns)
    else:
        cov_matrix = np.array([[np.var(asset_returns)]])

    # Perform Portfolio Optimization
    logger.info("Optimizing portfolio...")
    portfolio_optimizer = PortfolioOptimization(asset_returns, cov_matrix)
    optimized_results, optimal_weights = portfolio_optimizer.optimize_portfolio()

    # Visualize Metrics
    logger.info("Visualizing optimized portfolio...")
    asset_labels = ["Bonds", "Options", "Futures", "Swaps"]
    # Extract the first optimized portfolio from the results (assuming first row contains best weights)
    optimized_weights = optimized_results[0]  # Ensure correct slicing

    # Ensure optimized_weights is a 1D array of the right length
    optimized_weights = np.array(optimized_weights).flatten()

    # Check the shape and fix if necessary
    if len(optimized_weights) != len(asset_labels):
        optimized_weights = optimized_weights[:len(asset_labels)]  # Take only required number of assets

    # Plot the metrics
    plot_metrics(optimized_weights, asset_labels)


if __name__ == "__main__":
    main()
