import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
import time
import sys
import csv
from sklearn.model_selection import train_test_split
# distance Calculation
def dist(x, y, pos):
    return math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)

# Set Initial Parameters
areaSize = (10, 10)
node_pos = [(0, 0), (10, 0), (10, 10), (0, 10)]
possible_x = list(range(10, 90))
possible_y = list(range(10, 90))
num_particles = 200

# RSSI signal generation at pos(x,y) using path-loos model
def gen_wifi(freq=2.4,
             power=20,
             trans_gain=0,
             recv_gain=0,
             size=areaSize,
             pos=(5, 5),
             shadow_dev=1,
             n=2,
             rss0=-40,
             noise=1):
    if pos is None:
        pos = (random.randrange(size[0]), random.randrange(size[1]))
    random.seed(datetime.now())
    normal_dist = np.random.normal(0, shadow_dev, size=[size[0] + 1, size[1] + 1])
    rss = []
    random.seed(datetime.now())
    for x in range(0, 4):
        distance = dist(node_pos[x][0], node_pos[x][1], pos)
        val = rss0 - 10 * n * math.log10(distance) + normal_dist[int(pos[0])][int(pos[1])]
        rss.append(val - noise * random.random())
    return rss

#Load CSV Data
rssi_dict = []
for i in range(4):
    with open(sys.argv[1]+"s"+str(i)+".csv") as f:
        dict_from_csv = [{k: v for k, v in row.items()}
            for row in csv.DictReader(f,delimiter=';', skipinitialspace=True)]
    rssi_dict.append(dict_from_csv)

min_length = len(rssi_dict[0])
for i in range(1, 4):
    if len(rssi_dict[i]) < min_length:
        min_length = len(rssi_dict[i])

RSS0 = -47
overall_rss = []
original_tragectory = []
received_signal_log = []

for i in range(min_length):
    x, y = float(rssi_dict[0][i]['x']), float(rssi_dict[0][i]['y'])
    original_tragectory.append((x, y))
    random.seed(datetime.now())

    rss = [float(rssi_dict[0][i]['rssi']) - random.random(),
           float(rssi_dict[1][i]['rssi']) - random.random(),
           float(rssi_dict[2][i]['rssi']) - random.random(),
           float(rssi_dict[3][i]['rssi']) - random.random()]
    overall_rss.append(rss)


#dividing the dataset into train and test
Y = np.array(original_tragectory)
RSSI = np.array(overall_rss)
print(Y.shape)
print(RSSI.shape)

X = RSSI.reshape(RSSI.shape[0], 4)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

times = []
start_time = time.time()

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
regr = KNeighborsRegressor(n_neighbors=5,
                           algorithm='auto',
                           leaf_size=50,
                           p=2)

# 5-fold cross-validation
cross_val_scores = cross_val_score(regr, x_train, y_train, cv=5)
print("Cross-Validation Scores:", cross_val_scores)
regr .fit(x_train, y_train)

# Predict on new data
y_test_predict_r = regr.predict(x_test)
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

# #format conversion
# y_test_predict_r_first_column = []
# for sublist in y_test_predict_r:
#     y_test_predict_r_first_column.append(sublist[0])
# y_test_predict_r_second_column = []
# for sublist in y_test_predict_r:
#     y_test_predict_r_second_column.append(sublist[1])
# Y1 = []
# for sublist in Y:
#     Y1.append(sublist[0])
# Y2 = []
# for sublist in Y:
#     Y2.append(sublist[1])
#
# #Visualize robot movement trajectory and nodes
# plt.scatter(Y1, Y2, c='k')
# plt.scatter(y_test_predict_r_first_column, y_test_predict_r_second_column, marker='o', s=8)
# plt.plot(node_pos[0], node_pos[1], 'ro', markersize=5, clip_on=False, zorder=100)
# plt.plot(node_pos[2], node_pos[3], 'ro', markersize=5, clip_on=False, zorder=100)
# plt.legend(('real', 'predict', 'node'), fontsize='15')
# plt.ylim(-areaSize[1]+10, areaSize[1])
# plt.xlim(-areaSize[0]+10, areaSize[0])
# plt.title("Boundary Trajectory", fontsize='20')
# plt.ylabel('Y(m)')
# plt.xlabel('X(m)')
# plt.show()
