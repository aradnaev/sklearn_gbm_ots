import numpy as np


def remove_long_tail(df2, column, count_threshold, value):
    """Returns True if replaced values in column of dataframe
    with specified value
    if the value counts are below threshold."""
    v_counts = df2[column].value_counts()
    long_tail = v_counts[v_counts < count_threshold].index
    df_subset = df2[column].isin(long_tail)
    df2.loc[df_subset, column] = value
    return len(long_tail) > 0


def fill_na_median(df, col_name):
    df[col_name] = df[col_name].fillna(df[col_name].median())


def std(x, weights=None):
    # Generalized standard deviation with options of weights
    if weights is None:
        return np.std(x)
    else:
        weighted_mean = np.average(x, weights=weights)
        return np.sqrt(np.sum(((x - weighted_mean)**2) * weights)
                       / np.sum(weights))
