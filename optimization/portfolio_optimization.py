import numpy as np
class PortfolioOptimization:
    def __init__(self, returns, cov_matrix):
        self.returns = returns
        self.cov_matrix = cov_matrix

    def mean_variance_optimization(self, risk_free_rate=0.01):
        num_assets = len(self.returns)
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.dot(weights, self.returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return weights, portfolio_return, portfolio_volatility, sharpe_ratio

    def optimize_portfolio(self, num_portfolios=10000, risk_free_rate=0.01):
        results = np.zeros((3, num_portfolios))
        weights_record = []

        for i in range(num_portfolios):
            weights, portfolio_return, portfolio_volatility, sharpe_ratio = self.mean_variance_optimization(risk_free_rate)
            results[0, i] = portfolio_return
            results[1, i] = portfolio_volatility
            results[2, i] = sharpe_ratio
            weights_record.append(weights)

        return results, weights_record

    def get_optimal_weights(self, results, weights_record):
        max_sharpe_idx = np.argmax(results[2])
        return weights_record[max_sharpe_idx], results[0, max_sharpe_idx], results[1, max_sharpe_idx]