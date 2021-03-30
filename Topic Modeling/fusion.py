import pandas as pd

# fusion multiple csv files
def fusion(*args):
    data = pd.DataFrame()
    for data in args:
        data = pd.concat([data, pd.read_csv(arg)])
        data.to_csv("Topic_All_weeks.csv")
    return data