# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
log_data.append("Random Forest")
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

num_components = 20
pca = PCA(n_components=num_components)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
random_grid = {'n_estimators': [10, 20, 30, 40, 50, 100],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [4, 5, 6, 7, 8, 9, 10],
               'min_samples_split' : [2,4,8],
               'min_samples_leaf': [1, 5, 10, 20]}
grid_random_forest_performance = GridSearchCV(RandomForestRegressor(), param_grid = random_grid, scoring = 'r2', cv=5)
grid_random_forest_performance.fit(X_train_pca, y_train)
grid_random_forest_performance.best_params_

# %%
random_grid = {'n_estimators': [10, 20, 30, 40, 50, 100],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [4, 5, 6, 7, 8, 9, 10],
               'min_samples_split' : [2,4,8],
               'min_samples_leaf': [1, 5, 10, 20]}
grid_random_forest_performance = GridSearchCV(RandomForestRegressor(), param_grid = random_grid, scoring = 'r2', cv=5)
grid_random_forest_performance.fit(X_train_pca, y1_train)
grid_random_forest_performance.best_params_

# %%
random_grid = {'n_estimators': [10, 20, 30, 40, 50, 100],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [4, 5, 6, 7, 8, 9, 10],
               'min_samples_split' : [2,4,8],
               'min_samples_leaf': [1, 5, 10, 20]}
grid_random_forest_performance = GridSearchCV(RandomForestRegressor(), param_grid = random_grid, scoring = 'r2', cv=5)
grid_random_forest_performance.fit(X_train_pca, y2_train)
grid_random_forest_performance.best_params_

# %%
model = RandomForestRegressor(max_depth= 6, max_features= 'sqrt', min_samples_leaf= 20, min_samples_split= 2, n_estimators= 10)
model.fit(X_train_pca,y_train)

# %%
model2 = RandomForestRegressor(max_depth= 6, max_features= 'sqrt', min_samples_leaf= 5, min_samples_split= 4, n_estimators= 10)
model2.fit(X_train_pca,y1_train)

# %%
model3 = RandomForestRegressor(max_depth= 6, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 8, n_estimators= 10)
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


