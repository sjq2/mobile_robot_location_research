import math
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
import time


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
num_particles = 200
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

#Calculate Cooperative Direction of Arrival (CDOA) based on RSSI
prev = ()
doa = []
for i in range(0, len(overall_rss)):
    inner_curr = i
    limit = i-500 if i > 500 else 0
    est_sin_sum = 0
    est_cos_sum = 0
    starting_curr = inner_curr
    weight_sum = 0
    # average estimated DoA calculated
    while inner_curr >= limit:

        gx = ((overall_rss[i][1]-overall_rss[i][0])/2) + ((overall_rss[i][2]-overall_rss[i][3])/2)
        gy = ((overall_rss[i][1]-overall_rss[i][2])/2) + ((overall_rss[i][0]-overall_rss[i][3])/2)

        estimated_grad = np.arctan(gy/gx)
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

    if not prev:
        prev = (i, avg_grad)
    prev = (i, avg_grad)

#Select random particles
particles = []
for x in range(num_particles):
    particles.append((random.choice(possible_value), random.choice(possible_value)))

#Particle filtering computes the error
prev = ()
random.seed(datetime.now())

previous_errors = []
distance_error = []
times = []
position1 = []
Previous_pos = initial_pos
start_time = time.time()

for i in range(0, len(original_tragectory)):
    positions = []
    errors = []
    weights = []
    rands = []
    range_probs = []
    error = 0
    for particle in particles:
        x, y = particle[0], particle[1]
        actual_rss = gen_wifi(pos=(x, y), noise=0)
        gx = ((actual_rss[1]-actual_rss[0])/2) + ((actual_rss[2]-actual_rss[3])/2)
        gy = ((actual_rss[1]-actual_rss[2])/2) + ((actual_rss[0]-actual_rss[3])/2)
        adoa = np.arctan(gy/gx) if gx != 0 else 0
        error = abs(adoa-doa[i])

        if previous_errors:
            std_error = np.std(previous_errors)
        else:
            std_error = 0.001
        
        std_error = np.std(np.subtract(actual_rss, overall_rss[i]))
        omega = ((1/((std_error)*math.sqrt(2*math.pi)))*(math.pow(math.e, -(math.pow(error, 2)/(2*(std_error**2))))))

        for j in range(len(previous_errors)-1, len(previous_errors)-4 if len(previous_errors) > 5 else 0, -1):
            omega = omega*((1/((std_error)*math.sqrt(2*math.pi)))*(math.pow(math.e, -(math.pow(previous_errors[j], 2)/(2*(std_error**2))))))
    
        weights.append(omega)
        positions.append((x, y))
        errors.append(error)
        
    sum_weight = np.sum(weights)
    if sum_weight == 0:
        pass
    for j in range(0, len(weights)):
        weights[j] = weights[j]/sum_weight


    max_weight = max(weights)
    max_index = weights.index(max_weight)
    pos = positions[max_index]
    position1.append(pos)
    previous_errors.append(errors[max_index])
    distance_error.append(dist(pos[0], pos[1], original_tragectory[i]))

#Visualize robot movement trajectory and nodes
matrix2 = np.array(position1)
matrix = np.array(original_tragectory)

plt.plot(matrix[:, 0], matrix[:, 1], c='k')
plt.scatter(matrix2[:, 0], matrix2[:, 1], marker='o', s=8)
plt.plot(node_pos[0], node_pos[1], 'ro', markersize=5, clip_on=False, zorder=100)
plt.plot(node_pos[2], node_pos[3], 'ro', markersize=5, clip_on=False, zorder=100)
plt.ylim(-node_positions[1] - 4, node_positions[1] + 4)
plt.xlim(-node_positions[0] - 4, node_positions[0] + 4)
plt.legend(('original_tragectory', 'predict', 'node_positions'), fontsize='10')
plt.show()
# plt.savefig('PF_predicted_trajectory_boundry.png')

print("--- Computation Time: %s seconds ---" % (time.time() - start_time))
distcumulativeEror=np.sum(distance_error)/10
distmeanError=np.average(distance_error)/10
distStandardDeviationError=np.std(distance_error)/10
print("DIST_ERROR:   Cumulative Error: " + str(distcumulativeEror)+"\tMean  Error: "+str(distmeanError)+"\tStandard Deviation: "+str(distStandardDeviationError))
