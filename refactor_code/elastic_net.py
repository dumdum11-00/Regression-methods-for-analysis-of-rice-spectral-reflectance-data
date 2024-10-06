# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
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
log_data.append("elastic_net")
log_data.append("")


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
    X, Y, test_size=0.4, random_state=101)

X_train, X_test, y1_train, y1_test = train_test_split(
    X, Y1, test_size=0.4, random_state=101)

X_train, X_test, y2_train, y2_test = train_test_split(
    X, Y2, test_size=0.4, random_state=101)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)


# %%
# Create the random grid
random_grid = {"max_iter": [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100],
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                "l1_ratio": np.arange(0.0, 1.0, 0.1)}
grid_ridge_performance = GridSearchCV(ElasticNet(), param_grid = random_grid, scoring = 'r2')
grid_ridge_performance.fit(X_train_scaled, y_train)
grid_ridge_performance.best_params_

# %%
model = ElasticNet(alpha= 10, l1_ratio= 0.0, max_iter= 20)
model.fit(X_train_scaled,y_train)

model2 = ElasticNet(alpha= 10, l1_ratio= 0.7, max_iter= 10)
model2.fit(X_train_scaled,y1_train)

model3 = ElasticNet(alpha= 100, l1_ratio= 0.8, max_iter= 40)
model3.fit(X_train_scaled,y2_train)

# %%
predictions = model.predict(X_test_scaled)
predictions1 = model2.predict(X_test_scaled)
predictions2 = model3.predict(X_test_scaled)



# %%
chlo_score = r2_score(y_test, predictions)
chlor_mean_err = mean_squared_error(y_test, predictions)
log_data.append("R square of Chlorophyll-a: " +str(chlo_score))
log_data.append("Mean squared error of Chlorophyll-a: " +str(chlor_mean_err))

p_score = r2_score(y1_test, predictions1)
p_mean_err = mean_squared_error(y1_test, predictions1)
log_data.append("R square of P: " + str(p_score))
log_data.append("Mean squared error of P: " +str(p_mean_err))

k_score = r2_score(y2_test, predictions2)
k_mean_err = mean_squared_error(y2_test, predictions2)
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
chlo_mean_abs_err = mape(y_test, predictions)
p_mean_abs_err = mape(y1_test, predictions1)
k_mean_abs_err = mape(y2_test, predictions2)
log_data.append("Mean absolute percentage error of Chlorophyll-a: " +str(chlo_mean_abs_err))
log_data.append("Mean absolute percentage error of P: " +str(p_mean_abs_err))
log_data.append("Mean absolute percentage error of K: " +str(k_mean_abs_err))


print('Mean absolute percentage error of Chlorophyll-a: ', chlo_mean_abs_err)
print('Mean absolute percentage error of P: ', p_mean_abs_err)
print('Mean absolute percentage error of K: ', k_mean_abs_err)

# %%
log(log_data)


