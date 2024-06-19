import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression


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
    # dataframe to dictionaries
    categorical = ['PULocationID', 'DOLocationID']
    data_dicts = data[categorical].to_dict(orient='records')

    # transform variables for the model
    dv = DictVectorizer()
    X = dv.fit_transform(data_dicts)
    y = data.duration.values

    # train regression model
    model = LinearRegression()
    model.fit(X, y)

    return {'model': model, 'vectorizer': dv}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    print(output['model'].intercept_)
    assert output is not None, 'The output is undefined'