import unittest
from optimization.portfolio_optimization import PortfolioOptimization

class TestPortfolioOptimization(unittest.TestCase):

    def setUp(self):
        self.portfolio_optimizer = PortfolioOptimization()

    def test_mean_variance_optimization(self):
        # Example test for mean-variance optimization
        returns = [0.1, 0.2, 0.15]
        risks = [0.05, 0.1, 0.07]
        weights = self.portfolio_optimizer.mean_variance_optimization(returns, risks)
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_risk_adjustment(self):
        # Example test for risk adjustment
        portfolio_returns = [0.1, 0.2, 0.15]
        adjusted_risk = self.portfolio_optimizer.adjust_risk(portfolio_returns)
        self.assertIsInstance(adjusted_risk, float)

    def test_portfolio_metrics(self):
        # Example test for calculating portfolio metrics
        weights = [0.4, 0.3, 0.3]
        expected_return = self.portfolio_optimizer.calculate_expected_return(weights)
        self.assertGreater(expected_return, 0)

if __name__ == '__main__':
    unittest.main()