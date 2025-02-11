import unittest
from models.swap_pricing import SwapPricing

class TestSwapPricing(unittest.TestCase):

    def setUp(self):
        self.swap_pricing = SwapPricing()

    def test_present_value_of_swap(self):
        notional = 1000000
        fixed_rate = 0.02
        floating_rate = 0.015
        maturity = 5
        expected_value = self.swap_pricing.calculate_present_value(notional, fixed_rate, floating_rate, maturity)
        self.assertIsInstance(expected_value, float)

    def test_swap_cash_flows(self):
        notional = 1000000
        fixed_rate = 0.02
        floating_rate = 0.015
        maturity = 5
        cash_flows = self.swap_pricing.calculate_cash_flows(notional, fixed_rate, floating_rate, maturity)
        self.assertEqual(len(cash_flows), maturity)

if __name__ == '__main__':
    unittest.main()