import unittest
from models.futures_pricing import FuturesPricing

class TestFuturesPricing(unittest.TestCase):

    def setUp(self):
        self.futures_pricing = FuturesPricing()

    def test_calculate_futures_price(self):
        # Example test case for calculating futures price
        spot_price = 100
        strike_price = 105
        time_to_maturity = 30  # in days
        risk_free_rate = 0.01  # 1%
        expected_price = self.futures_pricing.calculate_futures_price(spot_price, strike_price, time_to_maturity, risk_free_rate)
        self.assertIsInstance(expected_price, float)

    def test_margin_requirements(self):
        # Example test case for margin requirements
        contract_value = 10000
        margin_percentage = 0.1  # 10%
        expected_margin = self.futures_pricing.calculate_margin_requirements(contract_value, margin_percentage)
        self.assertEqual(expected_margin, 1000)

    def test_rolling_adjustments(self):
        # Example test case for rolling adjustments
        current_price = 100
        new_price = 105
        adjustment = self.futures_pricing.calculate_rolling_adjustment(current_price, new_price)
        self.assertIsInstance(adjustment, float)

if __name__ == '__main__':
    unittest.main()