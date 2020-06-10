import pandas as pd

data = pd.read_csv("./data/golden_set.csv")
data["IS_BUG"] = data["classification"].apply(lambda x: 1 if x == 1 else 0)
data.to_csv("./data/golden_set_2.csv")