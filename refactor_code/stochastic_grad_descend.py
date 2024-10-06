# %%
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import SGDRegressor 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


# %%
from datetime import datetime

def log(msg):
    f = open("log.txt", "a")
    f.write("\n")
    f.write("==============================================")
    f.write("\n")
    for line in msg:
        f.write(line)
        f.write("\n")
    f.close()
    
log_data= [datetime.now().strftime("%d/%m/%Y, %H:%M:%S")]
log_data.append("stochastic_grad_descend")

# %%
df = pd.read_csv('processed_data.csv')
print(df.head(10))

# %%
scaler = StandardScaler()

X = df.iloc[: , 3:]

print(X)

Y = df['Chlorophyll-a']
Y1 = df['P conc. (mg/kg)']
Y2 = df['K conc. (mg/kg)']

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.4, random_state=42)

X_train, X_test, y1_train, y1_test = train_test_split(
    X, Y1, test_size=0.4, random_state=42)

X_train, X_test, y2_train, y2_test = train_test_split(
    X, Y2, test_size=0.4, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# %%
sgd_regressor = SGDRegressor(max_iter=1000, alpha=0.0001, learning_rate='invscaling', random_state=42) 

# %%
sgd_regressor.fit(X_train, y_train) 
y_predict = sgd_regressor.predict(X_test) 

# %%
sgd_regressor.fit(X_train, y1_train) 
y1_predict = sgd_regressor.predict(X_test) 

# %%
sgd_regressor.fit(X_train, y2_train) 
y2_predict = sgd_regressor.predict(X_test) 

# %%
chlo_score = r2_score(y_test, y_predict)
chlor_mean_err = mean_squared_error(y_test, y_predict)
log_data.append("R square of Chlorophyll-a: " +str(chlo_score))
log_data.append("Mean squared error of Chlorophyll-a: " +str(chlor_mean_err))

p_score = r2_score(y1_test, y1_predict)
p_mean_err = mean_squared_error(y1_test, y1_predict)
log_data.append("R square of P: " + str(p_score))
log_data.append("Mean squared error of P: " +str(p_mean_err))

k_score = r2_score(y2_test, y2_predict)
k_mean_err = mean_squared_error(y2_test, y2_predict)
log_data.append("R square of K: " + str(k_score))
log_data.append("Mean squared error of K: " +str(k_mean_err))


print('R square of Chlorophyll-a: ', chlo_score,)
print('R square of P: ', p_score,)
print('R square of K: ', k_score,)
print('Mean squared error of Chlorophyll-a: ', chlor_mean_err,)
print('Mean squared error of P: ', p_mean_err,)
print('Mean squared error of K: ', k_mean_err,)


# %%
def mape(y_test, pred):
    y_test, pred = np.array(y_test), np.array(pred)
    mape = np.mean(np.abs((y_test - pred) / y_test))
    return mape

# %%
chlo_mean_abs_err = mape(y_test, y_predict)
p_mean_abs_err = mape(y1_test, y1_predict)
k_mean_abs_err = mape(y2_test, y2_predict)
log_data.append("Mean absolute percentage error of Chlorophyll-a: " +str(chlo_mean_abs_err))
log_data.append("Mean absolute percentage error of P: " +str(p_mean_abs_err))
log_data.append("Mean absolute percentage error of K: " +str(k_mean_abs_err))


print('Mean absolute percentage error of Chlorophyll-a: ', chlo_mean_abs_err)
print('Mean absolute percentage error of P: ', p_mean_abs_err)
print('Mean absolute percentage error of K: ', k_mean_abs_err)

# %%
log(log_data)


