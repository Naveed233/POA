import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, fred_api_key, alpha_vantage_api_key, swap_api_key):
        """Initialize DataFetcher with API keys."""
        self.fred_api_key = fred_api_key
        self.alpha_vantage_api_key = alpha_vantage_api_key
        self.swap_api_key = swap_api_key  # ✅ FIXED: Now accepting SwapAPI Key

    def fetch_bond_yields(self):
        """Fetches bond yields from FRED API."""
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&api_key={self.fred_api_key}&file_type=json"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if "observations" not in data:
                logger.error("No observations found in response")
                return pd.DataFrame()

            df = pd.DataFrame(data["observations"])
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna()

            logger.info(f"Successfully fetched {len(df)} bond yield observations")
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching bond yields: {str(e)}")
            return pd.DataFrame()

    def fetch_options_data(self, symbol):
        """Fetches stock overview data from Alpha Vantage."""
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={self.alpha_vantage_api_key}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if "Error Message" in data:
                logger.error(f"Alpha Vantage API Error: {data['Error Message']}")
                return pd.DataFrame()

            return pd.DataFrame([data])
        except requests.exceptions.RequestException as e:
            logger.error(f"API Request failed: {str(e)}")
            return pd.DataFrame()

    def fetch_futures_data(self, symbol):
        """Fetches intraday time series data for a given stock symbol."""
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={self.alpha_vantage_api_key}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if "Time Series (1min)" not in data:
                logger.error("Invalid response: 'Time Series (1min)' key not found")
                return pd.DataFrame()

            return pd.DataFrame(data["Time Series (1min)"]).T
        except requests.exceptions.RequestException as e:
            logger.error(f"API Request failed: {str(e)}")
            return pd.DataFrame()

    def fetch_swap_rates(self):
        """Fetches swap rates from SwapAPI."""
        url = f"https://api.swapapi.com/v1/swaps?apikey={self.swap_api_key}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if "swaps" not in data:
                logger.error("Invalid swap API response: 'swaps' key missing")
                return pd.DataFrame()

            df = pd.DataFrame(data["swaps"])

            required_columns = ["currency_pair", "rate", "tenor", "timestamp"]
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"Missing column: {col} in swap data")

            logger.info(f"Successfully fetched {len(df)} swap rates")
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Swap API request failed: {e}")
            return pd.DataFrame()
