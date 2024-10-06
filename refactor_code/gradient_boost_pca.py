# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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
log_data.append("gradient_boost")
log_data.append("PCA")


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

# print(y1_train.isnull().values.any())

# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA()
pca.fit(X_train_scaled)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')

num_components = 3
pca = PCA(n_components=num_components)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
random_grid = {'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]}
grid_svr_performance = GridSearchCV(GradientBoostingRegressor(), param_grid = random_grid, scoring = 'r2', cv=5)
grid_svr_performance.fit(X_train_pca, y_train)
grid_svr_performance.best_params_

# %%

random_grid = {'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]}
grid_svr_performance = GridSearchCV(GradientBoostingRegressor(), param_grid = random_grid, scoring = 'r2', cv=5)
grid_svr_performance.fit(X_train_pca, y1_train)
grid_svr_performance.best_params_

# %%
random_grid = {'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2, 0.5, 1.0],
    'max_depth': [3, 4, 5, 6, 10]}
grid_svr_performance = GridSearchCV(GradientBoostingRegressor(), param_grid = random_grid, scoring = 'r2', cv=5)
grid_svr_performance.fit(X_train_pca, y2_train)
grid_svr_performance.best_params_

# %%
model = GradientBoostingRegressor(learning_rate=0.01, max_depth=3, n_estimators=100)
model.fit(X_train_pca,y_train)


# %%
model2 = GradientBoostingRegressor(learning_rate=0.01, max_depth=3, n_estimators=100)
model2.fit(X_train_pca,y1_train)

# %%
model3 = GradientBoostingRegressor(learning_rate=0.01, max_depth=3, n_estimators=100)
model3.fit(X_train_pca,y2_train)

# %%
predictions = model.predict(X_test_pca)
predictions1 = model2.predict(X_test_pca)
predictions2 = model3.predict(X_test_pca)

# %%
chlo_score = r2_score(y_test, predictions)
chlor_mean_err = mean_squared_error(y_test, predictions)

p_score = r2_score(y1_test, predictions1)
p_mean_err = mean_squared_error(y1_test, predictions1)

k_score = r2_score(y2_test, predictions2)
k_mean_err = mean_squared_error(y2_test, predictions2)

log_data.append("R square of Chlorophyll-a: " +str(chlo_score))
log_data.append("R square of P: " + str(p_score))
log_data.append("R square of K: " + str(k_score))
log_data.append("Mean squared error of Chlorophyll-a: " +str(chlor_mean_err))
log_data.append("Mean squared error of P: " +str(p_mean_err))
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
from csv import writer

csv_data = log_data[:3]

for i in log_data[3:]:
    i = i.split(": ")[1]
    csv_data.append(i)
    
with open('result.csv', 'a') as file:
    csvWriter = writer(file ,delimiter=',')
    csvWriter.writerow(csv_data)
    file.close()

log(log_data)


