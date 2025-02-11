import pandas as pd
import logging

logger = logging.getLogger(__name__)

def process_bond_data(bond_data):
    """Process bond yield data by converting percentages to decimal format."""
    if 'value' in bond_data.columns:
        bond_data['yield'] = bond_data['value'] / 100
    else:
        logger.warning("'value' column missing in bond data. Skipping yield calculation.")
    return bond_data

def process_derivative_data(derivative_data):
    """Process options, futures, and swaps data."""
    options_data = derivative_data.get("options", pd.DataFrame())
    futures_data = derivative_data.get("futures", pd.DataFrame())
    swaps_data = derivative_data.get("swaps", pd.DataFrame())

    # Process options data
    if not options_data.empty and "impliedVolatility" in options_data.columns:
        options_data["implied_volatility"] = options_data["impliedVolatility"].astype(float)

    # Process futures data
    if not futures_data.empty and "4. close" in futures_data.columns:
        futures_data["price"] = futures_data["4. close"].astype(float)

    # Process swaps data
    if not swaps_data.empty:
        swaps_data["rate"] = swaps_data["rate"].astype(float)
        swaps_data["tenor"] = swaps_data["tenor"].astype(int)

    return {
        "options": options_data,
        "futures": futures_data,
        "swaps": swaps_data,
    }

def transform_data(df):
    """Normalize values by standard deviation scaling."""
    if 'value' in df.columns:
        df['value'] = (df['value'] - df['value'].mean()) / df['value'].std()
    else:
        logger.warning("'value' column missing in data. Skipping normalization.")
    return df

def prepare_for_analysis(df):
    """Prepare data for analysis by setting the index to 'date'."""
    if 'date' in df.columns:
        df.set_index('date', inplace=True)
    else:
        logger.warning("'date' column missing in data. Skipping index setting.")
    return df

def aggregate_data(df, frequency='M'):
    """Aggregate data by the specified frequency (e.g., daily, monthly)."""
    if 'date' in df.index:
        return df.resample(frequency).mean()
    else:
        logger.warning("Data is not indexed by 'date'. Unable to resample.")
        return df
