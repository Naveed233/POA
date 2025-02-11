import unittest
from optimization.risk_adjustment import RiskAdjustment

class TestRiskAdjustment(unittest.TestCase):

    def setUp(self):
        self.risk_adjustment = RiskAdjustment()

    def test_variance_covariance_matrix(self):
        # Example data for testing
        returns = [[0.01, 0.02, 0.015], [0.02, 0.03, 0.025], [0.015, 0.025, 0.02]]
        expected_output = [[0.0001, 0.00015, 0.000125], [0.00015, 0.0002, 0.000175], [0.000125, 0.000175, 0.00015]]
        result = self.risk_adjustment.variance_covariance_matrix(returns)
        self.assertAlmostEqual(result, expected_output, places=5)

    def test_monte_carlo_simulation(self):
        # Example parameters for testing
        initial_investment = 100000
        num_simulations = 1000
        expected_shape = (1000, 10)  # Assuming 10 time steps
        result = self.risk_adjustment.monte_carlo_simulation(initial_investment, num_simulations)
        self.assertEqual(result.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()