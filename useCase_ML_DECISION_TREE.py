
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression


'''
Course:             FINS3648
Reference:          Raschka(2015)        
ML package links:   http://scikit-learn.org/stable/index.html
Data Source:        https://github.com/selva86/datasets/blob/master/BostonHousing.csv
Attributes:
        1. CRIM     per capita crime rate by town
        2. ZN       proportion of residential land zoned for lots over 
                    25,000 sq.ft.
        3. INDUS    proportion of non-retail business acres per town
        4. CHAS     Charles River dummy variable (= 1 if tract bounds 
                    river; 0 otherwise)
        5. NOX      nitric oxides concentration (parts per 10 million)
        6. RM       average number of rooms per dwelling
        7. AGE      proportion of owner-occupied units built prior to 1940
        8. DIS      weighted distances to five Boston employment centres
        9. RAD      index of accessibility to radial highways
        10. TAX     full-value property-tax rate per $10,000
        11. PTRATIO pupil-teacher ratio by town
        12. B       1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
                    by town
        13. LSTAT   % lower status of the population
        14. MEDV    Median value of owner-occupied homes in $1000's

DATA SAMPLE:   
CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	MEDV
0	0.00632	18	2.31	0	0.538	6.575	65.2	4.0900	1	296	15.3	396.90	4.98	24.0
1	0.02731	0	7.07	0	0.469	6.421	78.9	4.9671	2	242	17.8	396.90	9.14	21.6
2	0.02729	0	7.07	0	0.469	7.185	61.1	4.9671	2	242	17.8	392.83	4.03	34.7
3	0.03237	0	2.18	0	0.458	6.998	45.8	6.0622	3	222	18.7	394.63	2.94	33.4
4	0.06905	0	2.18	0	0.458	7.147	54.2	6.0622	3	222	18.7	396.90	5.33	36.2
'''

# Load Data Housing in Boston area
df = pd.read_csv("/Users/alinalimbu/Downloads/F3648/boston.csv")
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# define basic graphical funcional form
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='lightblue')
    plt.plot(X, model.predict(X), color='red', linewidth=2)
    return

from sklearn.tree import DecisionTreeRegressor

X = df[['RM']].values
y = df['MEDV'].values

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

sort_idx = X.flatten().argsort()

lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% lower status of the population [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.show()

