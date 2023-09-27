"""
plot_results



@Author: linlin
@Date: 31.05.23
"""

import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Datasets': ['Alkane', 'Acyclic', 'Redox ΔG_red^PBE0', 'Redox ΔG_ox^PBE0', 'MAO', 'PAH', 'MUTUG', 'Monoterpens', 'PTC_MR', 'Letter-high'],
    'Random': [13.4, 29.2, 25.3, 24.4, 80.0, 69.0, 80.0, 71.4, 56.3, 84.3],
    'Expert': [10.6, 30.4, 36.2, 40.0, 74.3, 71.0, 81.6, 71.7, 56.0, 84.3],
    'Target': [5.9, 15.0, 24.8, 26.8, 80.0, 68.0, 78.9, 70.7, 59.4, 91.8],
    'Path': [6.4, 13.0, 20.1, 26.4, 81.4, 68.0, 82.6, 71.0, 55.7, 90.1],
    'Treelet': [5.9, 16.8, 19.4, 25.8, 81.4, 74.0, 84.7, 70.0, 55.1, None],
    'WLSubtree': [8.2, 14.3, 22.1, 26.7, 84.3, 71.0, 81.1, 72.4, 60.0, None],
    'GCN': [7.4, 14.0, 26.1, 28.5, 84.3, 67.0, None, None, None, None],
    'GAT': [None, None, None, None, None, None, None, None, None, None]
}

df = pd.DataFrame(data)
df.set_index('Datasets', inplace=True)

df.plot(kind='bar', figsize=(10,7))
plt.ylabel('RMSE')
plt.title('Results on each dataset in terms of RMSE for the 10 splits, measured on the test sets')
plt.show()
