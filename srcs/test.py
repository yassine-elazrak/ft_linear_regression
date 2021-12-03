import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split , cross_val_score 
from sklearn import metrics
 

path = "data/data.csv"
df = pd.read_csv(path)
print(df.head())

# df.plot(x='km', y='price', style='*')
# plt.title('Square Feet vs Sale Price')
# plt.xlabel('Square Feet')
# plt.ylabel('Sale Price')
# plt.show()

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
def get_cv_scores(model):
    scores = cross_val_score(model,
                             X_train,
                             y_train,
                             cv=10,
                             scoring='r2')
    
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))
    print('\n')
    
lr = LinearRegression().fit(X_train, y_train)
get_cv_scores(lr)
print(lr.intercept_)
print(lr.coef_)

y_pred = lr.predict(X_test)
plt.scatter(X_train, y_train)
plt.plot(X_test, y_pred, color='red')
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))