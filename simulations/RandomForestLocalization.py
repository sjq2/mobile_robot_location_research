import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
import time
from sklearn.model_selection import train_test_split

# distance Calculation
def dist(x, y, pos):
    return math.sqrt((pos[0]-x)**2 + (pos[1]-y)**2)

# Simulation Environment variable Initialization
areaSize = (30, 30)
node_positions = (areaSize[0]+6, areaSize[1]+6)
node_pos = [(-node_positions[0], node_positions[1]),
            (node_positions[0], node_positions[1]),
            (node_positions[0], -node_positions[1]),
            (-node_positions[0], -node_positions[1])]
initial_pos = (0, 0)
possible_value = list(range(-30, 30))
NOISE_LEVEL = 1
RESOLUTION = 5
STEP_SIZE = 1/RESOLUTION

# RSSI signal generation at pos(x,y) using path-loos model
def gen_wifi(freq=2.4,
             power=20,
             trans_gain=0,
             recv_gain=0,
             size=areaSize,
             pos=(5, 5),
             shadow_dev=2,
             n=3,
             noise=NOISE_LEVEL):
    if pos is None:
        pos = (random.randrange(size[0]), random.randrange(size[1]))

    random.seed(datetime.now())
    rss0 = power + trans_gain + recv_gain + 20 * math.log10(3 / (4 * math.pi * freq * 10))
    rss0 = rss0-noise*random.random()
    normal_dist = np.random.normal(0, shadow_dev, size=[size[0]+1, size[1]+1])

    rss = []
    random.seed(datetime.now())

    for x in range(0, 4):
        distance = dist(node_pos[x][0], node_pos[x][1], pos)
        val = rss0 - 10 * n * math.log10(distance) + normal_dist[int(pos[0])][int(pos[1])]
        rss.append(val-noise*random.random())
    return rss

# Robot path trajectory generation
text = []
overall_rss = []
original_tragectory = []
Previous_pos = initial_pos

def move(pos):
    x, y = pos[0], pos[1]
    original_tragectory.append((x, y))
    rss = gen_wifi(pos=(x, y))
    overall_rss.append(rss)

y = areaSize[1]
for x in np.arange(0, areaSize[0], STEP_SIZE):
    move((x, y))
    Previous_pos = (x, y)
x = areaSize[0]
for y in np.arange(areaSize[1], -areaSize[1], -STEP_SIZE):
    move((x, y))
    Previous_pos = (x, y)
y = -areaSize[1]
for x in np.arange(areaSize[0], -areaSize[0], -STEP_SIZE):
    move((x, y))
    Previous_pos = (x, y)
x = -areaSize[0]
for y in np.arange(-areaSize[0], areaSize[0], STEP_SIZE):
    move((x, y))
    Previous_pos = (x, y)
y = areaSize[1]
for x in np.arange(-areaSize[0], 0, STEP_SIZE):
    move((x, y))
    Previous_pos = (x, y)

#dividing the dataset into train and test
Y = np.array(original_tragectory)
RSSI = np.array(overall_rss)
X = RSSI.reshape(1200, 4)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


times = []
start_time = time.time()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)


# 5-fold cross-validation
cross_val_scores = cross_val_score(rfr, x_train, y_train, cv=5)
print("Cross-Validation Scores:", cross_val_scores)
rfr .fit(x_train, y_train)

# Predict on new data
y_test_predict_r = rfr.predict(x_test)


Y_test = y_test.tolist()
Y_test_predict_r = y_test_predict_r.tolist()

#calculate prediction erro
distance_error_test = []
for i in range(0, len(Y_test)):
    p_test = Y_test_predict_r[i]
    t_test = Y_test[i]
    distance_error_test.append(dist(p_test[0], p_test[1], t_test))
distcumulativeEror_test = np.sum(distance_error_test)/10
distmeanError_test = np.average(distance_error_test)/10
distStandardDeviationError_test = np.std(distance_error_test)/10

print("test_ERROR:   Cumulative Error: " + str(distcumulativeEror_test)+"\tMean  Error: "+str(distmeanError_test)+"\tStandard Deviation: "+str(distStandardDeviationError_test))
print("--- Computation Time: %s seconds ---" % (time.time() - start_time))

#format conversion
y_test_predict_r_first_column = []
for sublist in y_test_predict_r:
    y_test_predict_r_first_column.append(sublist[0])
y_test_predict_r_second_column = []
for sublist in y_test_predict_r:
    y_test_predict_r_second_column.append(sublist[1])
Y1 = []
for sublist in Y:
    Y1.append(sublist[0])
Y2 = []
for sublist in Y:
    Y2.append(sublist[1])

#Visualize robot movement trajectory and nodes
plt.plot(Y1, Y2, c='k')
plt.scatter(y_test_predict_r_first_column, y_test_predict_r_second_column, marker='o', s=8)
plt.plot(node_pos[0], node_pos[1], 'ro', markersize=5, clip_on=False, zorder=100)
plt.plot(node_pos[2], node_pos[3], 'ro', markersize=5, clip_on=False, zorder=100)
plt.legend(('real', 'predict', 'node'), fontsize='15')
plt.title("Boundary Trajectory", fontsize='20')
plt.ylabel('Y(m)')
plt.xlabel('X(m)')
plt.show()