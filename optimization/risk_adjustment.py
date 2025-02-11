class RiskAdjustment:
    def __init__(self, returns):
        self.returns = returns

    def calculate_variance_covariance_matrix(self):
        return self.returns.cov()

    def monte_carlo_simulation(self, num_simulations, horizon):
        simulated_returns = np.random.normal(
            loc=self.returns.mean(), 
            scale=self.returns.std(), 
            size=(num_simulations, horizon)
        )
        cumulative_returns = np.exp(np.sum(simulated_returns, axis=1))
        return cumulative_returns

    def value_at_risk(self, confidence_level=0.95):
        return np.percentile(self.returns, (1 - confidence_level) * 100)

    def expected_shortfall(self, confidence_level=0.95):
        var = self.value_at_risk(confidence_level)
        return self.returns[self.returns <= var].mean()