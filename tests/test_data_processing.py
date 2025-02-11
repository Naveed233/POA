import pytest
import pandas as pd
from data.data_processing import clean_data, transform_data

def test_clean_data():
    raw_data = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-02', None],
        'value': ['1.5', '2.0', '3.0']
    })
    cleaned_data = clean_data(raw_data)
    assert cleaned_data.shape[0] == 2
    assert 'date' in cleaned_data.columns
    assert 'value' in cleaned_data.columns
    assert cleaned_data['value'].dtype == float

def test_transform_data():
    raw_data = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
        'value': [1.5, 2.0]
    })
    transformed_data = transform_data(raw_data)
    assert 'transformed_value' in transformed_data.columns
    assert transformed_data['transformed_value'].equals(pd.Series([1.5, 2.0]))  # Example transformation check