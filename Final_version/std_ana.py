import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import boxcox1p
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler as ss

# Generate some data for this
# demonstration.
two = pd.read_csv("two_species.csv", encoding="gbk")
t4 = two.loc[two["Month"] == 4]
#t4 = t4.loc[t4["Q"] != 0]
#bc = PowerTransformer(method="box-cox")

data = np.log(t4["Q"]+1)
#data = ss.fit_transform(t4)
# Fit a normal distribution to
# the data:
# mean and standard deviation
mu, std = norm.fit(data)

# Plot the histogram.
plt.hist(data, bins=25, density=True, alpha=0.6, color='b')

# Plot the PDF.
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)

plt.plot(x, p, 'k', linewidth=2)
title = "Mean:{:.2f} and Std_err:{:.2f}".format(mu, std)
plt.title(title)

plt.show()