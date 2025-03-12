import pandas as pd
import matplotlib.pyplot as plt
url = 'airline-passengers.csv'
df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
plt.plot(df)
plt.show()