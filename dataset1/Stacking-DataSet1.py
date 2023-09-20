import math
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
import time
import csv
import sys

# distance Calculation
def dist(x, y, pos):
    return math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)

#Load CSV train_Data
with open(sys.argv[1]) as f:
    dict_from_csv_train = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]

overall_rss_train = []
original_tragectory_train = []

for i in range(len(dict_from_csv_train)):
    dict_train = dict_from_csv_train[i]
    x, y = float(dict_train['x']), float(dict_train['y'])
    original_tragectory_train.append((x, y))
    random.seed(datetime.now())
    rss_train = [-int(float(dict_train['RSSI A'])) - random.random(),
                 -int(float(dict_train['RSSI B'])) - random.random(),
                 -int(float(dict_train['RSSI C'])) - random.random()]
    overall_rss_train.append(rss_train)

#Calculate Cooperative Direction of Arrival (CDOA) based on train_RSSI
doa_train = []
for i in range(0, len(overall_rss_train)):
    inner_curr = i
    limit = i - 500 if i > 500 else 0
    est_sin_sum = 0
    est_cos_sum = 0
    starting_curr = inner_curr
    weight_sum = 0
    # average estimated DoA calculated
    while inner_curr >= limit:

        gx = ((overall_rss_train[i][2] - overall_rss_train[i][0]) / 4)
        gy = ((overall_rss_train[i][1] - overall_rss_train[i][0]) / 4)

        estimated_grad = np.arctan(gy / gx)
        if estimated_grad > math.pi:
            estimated_grad = -2 * math.pi + estimated_grad
        elif estimated_grad < -math.pi:
            estimated_grad = math.pi - abs(-math.pi - estimated_grad)
        weight = 0.99 ** (inner_curr - starting_curr)
        weight_sum += weight
        estimated_grad = weight * estimated_grad
        est_sin_sum += math.sin(estimated_grad)
        est_cos_sum += math.cos(estimated_grad)
        inner_curr -= 1
    avg_est_sin = est_sin_sum / weight_sum
    avg_est_cos = est_cos_sum / weight_sum
    avg_grad = math.atan2(avg_est_sin, avg_est_cos)
    doa_train.append(avg_grad)

y_train = np.array(original_tragectory_train)
RSSI_train = np.array(overall_rss_train)
x_train = RSSI_train.reshape(49, 3)

#Load CSV test_Data
with open(sys.argv[2]) as f:
    dict_from_csv_test = [{k: v for k, v in row.items()}
        for row in csv.DictReader(f, skipinitialspace=True)]

overall_rss_test = []
original_tragectory_test = []

for i in range(len(dict_from_csv_test)):
    dict_test = dict_from_csv_test[i]
    x, y = float(dict_test['x']), float(dict_test['y'])
    original_tragectory_test.append((x, y))
    random.seed(datetime.now())
    rss_test = [-int(float(dict_test['RSSI A'])) - random.random(),
                -int(float(dict_test['RSSI B'])) - random.random(),
                -int(float(dict_test['RSSI C'])) - random.random()]
    overall_rss_test.append(rss_test)

#Calculate Cooperative Direction of Arrival (CDOA) based on test_RSSI
doa_test = []
for i in range(0, len(overall_rss_test)):
    inner_curr = i
    limit = i - 500 if i > 500 else 0
    est_sin_sum = 0
    est_cos_sum = 0
    starting_curr = inner_curr
    weight_sum = 0
    # average estimated DoA calculated
    while inner_curr >= limit:

        gx = ((overall_rss_test[i][2] - overall_rss_test[i][0]) / 4)
        gy = ((overall_rss_test[i][1] - overall_rss_test[i][0]) / 4)

        estimated_grad = np.arctan(gy / gx)
        if estimated_grad > math.pi:
            estimated_grad = -2 * math.pi + estimated_grad
        elif estimated_grad < -math.pi:
            estimated_grad = math.pi - abs(-math.pi - estimated_grad)
        weight = 0.99 ** (inner_curr - starting_curr)
        weight_sum += weight
        estimated_grad = weight * estimated_grad
        est_sin_sum += math.sin(estimated_grad)
        est_cos_sum += math.cos(estimated_grad)
        inner_curr -= 1
    avg_est_sin = est_sin_sum / weight_sum
    avg_est_cos = est_cos_sum / weight_sum
    avg_grad = math.atan2(avg_est_sin, avg_est_cos)
    doa_test.append(avg_grad)

y_test = np.array(original_tragectory_test)
RSSI_test = np.array(overall_rss_test)
x_test = RSSI_test.reshape(10, 3)


times = []
start_time = time.time()

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor


estimators = [('Random Forest', RandomForestRegressor(random_state=42)),
              ('r', LinearRegression()),
              ('knn', KNeighborsRegressor())
              ]

stackingRegressor = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())
mtl_rfr = MultiOutputRegressor(stackingRegressor)

# 5-fold cross-validation
cross_val_scores = cross_val_score(mtl_rfr, x_train, y_train, cv=5)
print("Cross-Validation Scores:", cross_val_scores)
mtl_rfr .fit(x_train, y_train)

# Predict on new data
y_train_predict_r = mtl_rfr.predict(x_train)
y_test_predict_r = mtl_rfr.predict(x_test)

Y_train = y_train.tolist()
Y_train_predict_r = y_train_predict_r.tolist()
Y_test = y_test.tolist()
Y_test_predict_r = y_test_predict_r.tolist()


#calculate prediction erro
distance_error_test = []
for i in range(0, len(Y_test)):
    p_test = Y_test_predict_r[i]
    t_test = Y_test[i]
    distance_error_test.append(dist(p_test[0], p_test[1], t_test))
distcumulativeEror_test = np.sum(distance_error_test)
distmeanError_test = np.average(distance_error_test)
distStandardDeviationError_test = np.std(distance_error_test)

print("test_ERROR:   Cumulative Error: " + str(distcumulativeEror_test)+"\tMean  Error: "+str(distmeanError_test)+"\tStandard Deviation: "+str(distStandardDeviationError_test))
print("--- Computation Time: %s seconds ---" % (time.time() - start_time))