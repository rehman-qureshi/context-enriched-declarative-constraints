import random
import math
import itertools
import pandas as pd
#------------------------------
def conditional_shapley_with_binning(
    df,
    target_col="Outcome",
    target_value="Violated",
    num_bins=5
):
    df = df.copy()

    # Step 1: Identify features
    features = [c for c in df.columns if c != target_col]

    # Step 2: Automatically bin numeric columns
    binned_features = []
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):# and col != "case:Item":
            new_col = col + "_bin"
            df[new_col] = pd.qcut(df[col], q=num_bins, duplicates="drop")
            #print(f"Binned column {col} into {num_bins} bins as {new_col}")
            #print(df[[col, new_col]].head())
            binned_features.append(new_col)
        else:
            binned_features.append(col)

    features = binned_features
    n = len(features)

    # Step 3: Cache for value function
    value_cache = {}

    def v(S):
        key = tuple(sorted(S))
        if key in value_cache:
            return value_cache[key]

        if len(S) == 0:
            val = (df[target_col] == target_value).mean()
        else:
            grouped = df.groupby(list(S),observed=False)[target_col].apply(
            lambda x: (x == target_value).mean())
            val = grouped.mean()
            
        value_cache[key] = val
        return val

    # Step 4: Compute Shapley values
    shapley = {f: 0 for f in features}

    for feature in features:
        others = [f for f in features if f != feature]

        for r in range(len(others) + 1):
            for S in itertools.combinations(others, r):

                S = set(S)
                S_with_i = S | {feature}

                weight = (
                    math.factorial(len(S)) *
                    math.factorial(n - len(S) - 1) /
                    math.factorial(n)
                )

                marginal = v(S_with_i) - v(S)
                shapley[feature] += weight * marginal

    return shapley
#------------------------------
