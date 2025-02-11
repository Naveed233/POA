import math
from scipy.stats import norm

class OptionPricing:
    def __init__(self, spot_price, strike_price, time_to_maturity, risk_free_rate, volatility):
        """Initialize option pricing model with Black-Scholes and Binomial Tree models."""
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility

    def black_scholes_call(self):
        """Calculates the Black-Scholes price for a call option."""
        d1 = (math.log(self.spot_price / self.strike_price) + 
              (self.risk_free_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / \
              (self.volatility * math.sqrt(self.time_to_maturity))
        d2 = d1 - self.volatility * math.sqrt(self.time_to_maturity)

        call_price = (self.spot_price * norm.cdf(d1) - 
                      self.strike_price * math.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(d2))
        return call_price

    def black_scholes_put(self):
        """Calculates the Black-Scholes price for a put option."""
        d1 = (math.log(self.spot_price / self.strike_price) + 
              (self.risk_free_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / \
              (self.volatility * math.sqrt(self.time_to_maturity))
        d2 = d1 - self.volatility * math.sqrt(self.time_to_maturity)

        put_price = (self.strike_price * math.exp(-self.risk_free_rate * self.time_to_maturity) * norm.cdf(-d2) - 
                     self.spot_price * norm.cdf(-d1))
        return put_price

    def binomial_tree_option(self, steps=100, option_type='call'):
        """Calculates the option price using the binomial tree model."""
        dt = self.time_to_maturity / steps
        u = math.exp(self.volatility * math.sqrt(dt))
        d = 1 / u
        p = (math.exp(self.risk_free_rate * dt) - d) / (u - d)

        # Initialize asset prices at maturity
        asset_prices = [self.spot_price * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]
        option_values = [max(0, price - self.strike_price) if option_type == 'call' else max(0, self.strike_price - price) for price in asset_prices]

        # Backward induction
        for i in range(steps - 1, -1, -1):
            option_values = [math.exp(-self.risk_free_rate * dt) * (p * option_values[j + 1] + (1 - p) * option_values[j]) for j in range(i + 1)]

        return option_values[0]
