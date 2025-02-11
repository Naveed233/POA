# Portfolio Optimization Application

This project is a comprehensive portfolio optimization application designed to support American bonds and derivatives, including options, futures, and swaps. It provides tools for data handling, pricing models, risk adjustments, and optimization features, along with visual analysis metrics.

## Project Structure

```
portfolio-optimization-app
├── data
│   ├── __init__.py
│   ├── data_fetching.py
│   └── data_processing.py
├── models
│   ├── __init__.py
│   ├── bond_pricing.py
│   ├── option_pricing.py
│   ├── futures_pricing.py
│   └── swap_pricing.py
├── optimization
│   ├── __init__.py
│   ├── portfolio_optimization.py
│   └── risk_adjustment.py
├── visualization
│   ├── __init__.py
│   └── visual_analysis.py
├── tests
│   ├── __init__.py
│   ├── test_data_fetching.py
│   ├── test_data_processing.py
│   ├── test_bond_pricing.py
│   ├── test_option_pricing.py
│   ├── test_futures_pricing.py
│   ├── test_swap_pricing.py
│   ├── test_portfolio_optimization.py
│   └── test_risk_adjustment.py
├── main.py
└── README.md
```

## Features

- **Data Fetching**: Fetches bond yields, options data, futures data, and swap rates.
- **Pricing Models**: Implements pricing models for bonds, options, futures, and swaps.
- **Portfolio Optimization**: Provides mean-variance optimization methods tailored for bonds and derivatives.
- **Risk Adjustment**: Calculates risk metrics and performs Monte Carlo simulations.
- **Visualization**: Visualizes portfolio metrics, including return distributions and risk metrics.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd portfolio-optimization-app
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:
```
python main.py
```

## Examples

Refer to the documentation within each module for specific usage examples and detailed explanations of the implemented methods.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.