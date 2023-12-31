import math
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
import time
import sys
import csv
import pandas as pd
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
path_loss_list = []
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
    if float(rssi_dict[0][i]['distance']) > 1.4 and float(rssi_dict[0][i]['distance']) < 1.5:
        RSS0 = float(rssi_dict[0][i]['rssi'])
    elif float(rssi_dict[1][i]['distance']) > 1.4 and float(rssi_dict[1][i]['distance']) < 1.5:
        RSS0 = float(rssi_dict[1][i]['rssi'])
    elif float(rssi_dict[2][i]['distance']) > 1.4 and float(rssi_dict[2][i]['distance']) < 1.5:
        RSS0 = float(rssi_dict[2][i]['rssi'])
    elif float(rssi_dict[3][i]['distance']) > 1.4 and float(rssi_dict[3][i]['distance']) < 1.5:
        RSS0 = float(rssi_dict[3][i]['rssi'])
    for j in range(4):
        path_loss_list.append(20 - rss[j])
        received_signal_log.append(10 * math.log10(float(rssi_dict[j][i]['distance'])))


average_path_loss = np.average(path_loss_list)
average_received_signal_log = np.average(received_signal_log)
nominator = 0
demonimator = 0

for i in range(len(path_loss_list)):
    nominator += (path_loss_list[i] - average_path_loss) * (received_signal_log[i] - average_received_signal_log)
    demonimator += math.pow((received_signal_log[i] - average_received_signal_log), 2)
pathloss_exponent = nominator / demonimator

#Calculate Cooperative Direction of Arrival (CDOA) based on RSSI
doa = []
for i in range(0, len(overall_rss)):
    inner_curr = i
    limit = i - 500 if i > 500 else 0
    est_sin_sum = 0
    est_cos_sum = 0
    starting_curr = inner_curr
    weight_sum = 0
    # average estimated DoA calculated
    while inner_curr >= limit:
        gx = ((overall_rss[i][1] - overall_rss[i][0]) / 2) + ((overall_rss[i][2] - overall_rss[i][3]) / 2)
        gy = ((overall_rss[i][2] - overall_rss[i][1]) / 2) + ((overall_rss[i][3] - overall_rss[i][0]) / 2)
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
    doa.append(avg_grad)

#Particle filtering computes the error
random.seed(datetime.now())
previous_errors = []
distance_error = []
position_weight_list = []
particles = []
times = []

for i in range(num_particles):
    particles.append((random.choice(possible_x) / 10, random.choice(possible_y) / 10))
num_particles = len(particles)
print("Number of particle filters", num_particles)
position1 = []
for i in range(0, len(original_tragectory)):
    start_time = time.time()
    positions = []
    errors = []
    weights = []
    rands = []
    range_probs = []
    error = 0
    for particle in particles:
        x, y = particle[0], particle[1]
        actual_rss = gen_wifi(pos=(x, y), n=pathloss_exponent, rss0=RSS0, noise=0)
        gx = ((actual_rss[1] - actual_rss[0]) / 2) + ((actual_rss[2] - actual_rss[3]) / 2)
        gy = ((actual_rss[2] - actual_rss[1]) / 2) + ((actual_rss[3] - actual_rss[0]) / 2)

        adoa = np.arctan(gy / gx) if gx != 0 else 0
        error = abs(adoa - doa[i])

        std_error = np.std(np.subtract(actual_rss, overall_rss[i]))
        omega = ((1 / ((std_error) * math.sqrt(2 * math.pi))) * (
            math.pow(math.e, -(math.pow(error, 2) / (2 * (std_error ** 2))))))
        for j in range(len(previous_errors) - 1, len(previous_errors) - 4 if len(previous_errors) > 5 else 0, -1):
            omega = omega * ((1 / ((std_error) * math.sqrt(2 * math.pi))) * (
                math.pow(math.e, -(math.pow(previous_errors[j], 2) / (2 * (std_error ** 2))))))

        weights.append(omega)
        positions.append((x, y))
        errors.append(error)

    sum_weight = np.sum(weights)
    if sum_weight == 0:
        pass
    for j in range(0, len(weights)):
        weights[j] = weights[j] / sum_weight

    max_weight = max(weights)
    max_index = weights.index(max_weight)
    pos = positions[max_index]
    position1.append(pos)
    previous_errors.append(errors[max_index])
    distance_error.append(dist(pos[0], pos[1], original_tragectory[i]))
    times.append(time.time() - start_time)

distcumulativeEror = np.sum(distance_error)
distmeanError = np.average(distance_error)
distStandardDeviationError = np.std(distance_error)
#
# #Visualize robot movement trajectory and nodes
# matrix2 = np.array(position1)
# matrix = np.array(original_tragectory)
# plt.scatter(matrix[:, 0], matrix[:, 1], c='k')
# plt.scatter(matrix2[:, 0], matrix2[:, 1], marker='o', s=8)
# plt.plot([node_pos[0][0], node_pos[1][0], node_pos[2][0]], [node_pos[0][1], node_pos[1][1], node_pos[2][1]], 'ro', markersize=5, clip_on=False, zorder=100)
# plt.ylim(-areaSize[1]+10, areaSize[1])
# plt.xlim(-areaSize[0]+10, areaSize[0])
# plt.legend(('original', 'predict', 'node_positions'), fontsize='10')
# plt.show()
#

print("--- Average Computation Time per Iteration : %s seconds ---" % (np.average(times)))
print("rss0",RSS0,"path loss exponent: ",pathloss_exponent)

# print("RSS_ERROR:   Cumulative Error: " + str(rsscumulativeEror)+"\tMean  Error: "+str(rssmeanError)+"\tStandard Deviation: "+str(rssStandardDeviationError))
print("DIST_ERROR:   Cummulative Error: " + str(distcumulativeEror)+"\tMean  Error: "+str(distmeanError)+"\tStandard Deviation: "+str(distStandardDeviationError))
