import pandas as pd
from sklearn.linear_model import LinearRegression
from mlops.utils.data_preparation.encoders import vectorize_features
from mlops.utils.models.sklearn import train_model


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # transform variables for the model
    categorical = ['PULocationID', 'DOLocationID']
    X, _, dv = vectorize_features(data[categorical])
    y = data.duration.values

    # train regression model
    model, _, _ = train_model(LinearRegression(), X, y)

    return {'model': model, 'vectorizer': dv}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    print(output['model'].intercept_)
    assert output is not None, 'The output is undefined'