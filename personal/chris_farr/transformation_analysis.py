import random
import pandas as pd
import numpy as np

# Do features from non-linear transformations highly correlate with the original feature?

# Create a range of random numbers
# rand_ints = [random.randint(0, 100) for _ in range(100)]
rand_floats = [random.random() for _ in range(100)]
# Create pandas data frame
# df = pd.DataFrame(data=rand_ints, columns=["orig"])
df = pd.DataFrame(data=rand_floats, columns=["orig"])
# Perform non-linear transformations
df.loc[:, "log"] = df["orig"].apply(np.log)
# df.loc[:, "log2"] = df["orig"].apply(np.log2)
# df.loc[:, "log10"] = df["orig"].apply(np.log10)
# df.loc[:, "cubert"] = df["orig"].apply(lambda x: np.power(x, 1 / 3))
df.loc[:, "sqrt"] = df["orig"].apply(np.sqrt)
# df.loc[:, "exp"] = df["orig"].apply(np.exp)
# df.loc[:, "exp2"] = df["orig"].apply(np.exp2)
df.loc[:, "cube"] = df["orig"].apply(lambda x: np.power(x, 3))
df.loc[:, "square"] = df["orig"].apply(np.square)

# Calculate correlation matrix
corr_df = df.corr()

# Hypothesis: some will and some won't depending on the original values
# After repeated runs these are the transformation
# Final for int range(0, 100): log, sqrt, exp, cube, square
# Final for float range(0, 1): log, sqrt, cube, square
