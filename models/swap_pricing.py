import pandas as pd
import logging
import requests 

logger = logging.getLogger(__name__)

class SwapPricing:
    def __init__(self, swap_data):
        """Initialize Swap Pricing Model with swap market data."""
        self.swap_data = swap_data
        self.discount_rate = self.fetch_discount_rate()

    def fetch_discount_rate(self):
        """Fetches the U.S. Treasury average interest rate as a proxy for the risk-free rate."""
        url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if "data" not in data or not data["data"]:
                logger.warning("No interest rate data found. Using default 3% discount rate.")
                return 0.03  # Default 3% if data is unavailable

            # Get the latest interest rate record
            latest_rate = float(data["data"][0]["avg_interest_rate_amt"]) / 100  # Convert percentage to decimal
            logger.info(f"Fetched latest discount rate: {latest_rate:.4f}")
            return latest_rate
        except Exception as e:
            logger.error(f"Failed to fetch interest rate: {e}")
            return 0.03  # Default 3%

    def calculate_fixed_leg(self, notional, fixed_rate, maturity):
        """Calculates fixed leg payments (Present Value of fixed rate leg)."""
        return notional * fixed_rate * maturity

    def calculate_floating_leg(self, notional, floating_rates):
        """Calculates floating leg payments (PV of floating rate leg)."""
        floating_leg = sum(notional * rate for rate in floating_rates)
        return floating_leg

    def net_present_value(self, notional, fixed_rate, floating_rates, maturity):
        """Calculates the Net Present Value (NPV) of the swap contract."""
        fixed_leg_npv = self.calculate_fixed_leg(notional, fixed_rate, maturity) / (1 + self.discount_rate) ** maturity
        floating_leg_npv = self.calculate_floating_leg(notional, floating_rates) / (1 + self.discount_rate) ** maturity
        return fixed_leg_npv - floating_leg_npv

    def calculate_prices(self):
        """Calculates the swap contract values for different swap rates."""
        if self.swap_data.empty:
            logger.warning("No swap data available. Returning default swap price of 0.0")
            return 0.0  # Ensure float return to avoid TypeError

        swap_prices = {}

        for _, row in self.swap_data.iterrows():
            try:
                notional = 1_000_000  # Assuming 1M USD per contract
                fixed_rate = row.get("rate", 0.02)  # Default 2% if missing
                maturity = row.get("tenor", 5)  # Default 5 years
                floating_rates = [fixed_rate]  # Using the fixed rate for now

                pv = self.net_present_value(notional, fixed_rate, floating_rates, maturity)
                swap_prices[row.get("currency_pair", "USD")] = pv

            except Exception as e:
                logger.error(f"Error calculating swap price: {e}")

        return swap_prices if swap_prices else 0.0  # Ensure float return for portfolio calculations
