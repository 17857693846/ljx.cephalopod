import pandas as pd
import pandas_profiling as ppf

data = pd.read_csv("two_species.csv", encoding="gbk")
d = data.loc[data["Month"]==4]
p1 = ppf.ProfileReport(d)
p1.to_file("Spring Report")

d2 = data.loc[data["Month"]==11]
p2 = ppf.ProfileReport(d)
p2.to_file("Autumn Report")

