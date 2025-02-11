import unittest
from models.option_pricing import OptionPricing

class TestOptionPricing(unittest.TestCase):
    def setUp(self):
        self.option_pricing = OptionPricing()

    def test_black_scholes_call_price(self):
        # Test parameters for Black-Scholes call pricing
        S = 100  # Current stock price
        K = 100  # Strike price
        T = 1    # Time to expiration in years
        r = 0.05 # Risk-free interest rate
        sigma = 0.2 # Volatility

        call_price = self.option_pricing.black_scholes_call(S, K, T, r, sigma)
        self.assertIsInstance(call_price, float)

    def test_black_scholes_put_price(self):
        # Test parameters for Black-Scholes put pricing
        S = 100  # Current stock price
        K = 100  # Strike price
        T = 1    # Time to expiration in years
        r = 0.05 # Risk-free interest rate
        sigma = 0.2 # Volatility

        put_price = self.option_pricing.black_scholes_put(S, K, T, r, sigma)
        self.assertIsInstance(put_price, float)

    def test_binomial_tree_option_price(self):
        # Test parameters for Binomial Tree option pricing
        S = 100  # Current stock price
        K = 100  # Strike price
        T = 1    # Time to expiration in years
        r = 0.05 # Risk-free interest rate
        sigma = 0.2 # Volatility
        n = 100  # Number of steps

        binomial_price = self.option_pricing.binomial_tree_option_price(S, K, T, r, sigma, n)
        self.assertIsInstance(binomial_price, float)

if __name__ == '__main__':
    unittest.main()