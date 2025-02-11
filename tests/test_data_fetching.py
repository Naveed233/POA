import unittest
from data.data_fetching import DataFetcher

class TestDataFetcher(unittest.TestCase):
    def setUp(self):
        self.fred_api_key = 'test_fred_api_key'
        self.alpha_vantage_api_key = 'test_alpha_vantage_api_key'
        self.data_fetcher = DataFetcher(self.fred_api_key, self.alpha_vantage_api_key)

    def test_fetch_bond_yields(self):
        df = self.data_fetcher.fetch_bond_yields()
        self.assertFalse(df.empty)
        self.assertIn('date', df.columns)
        self.assertIn('value', df.columns)

    def test_fetch_options_data(self):
        symbol = 'AAPL'
        df = self.data_fetcher.fetch_options_data(symbol)
        self.assertFalse(df.empty)
        self.assertIn('strikePrice', df.columns)

    def test_fetch_futures_data(self):
        symbol = 'CL=F'
        df = self.data_fetcher.fetch_futures_data(symbol)
        self.assertFalse(df.empty)
        self.assertIn('1. open', df.columns)

    def test_fetch_swap_rates(self):
        # Placeholder test for swap rates
        self.assertIsNone(self.data_fetcher.fetch_swap_rates())

if __name__ == '__main__':
    unittest.main()