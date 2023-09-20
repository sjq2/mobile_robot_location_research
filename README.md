# mobile_robot_location_research
## 仿真实验
对**6×6m**的有界区域进行仿真实验，在机器人工作空间的四个角落(0,0)、(0,6)、(6,0)、(6,6)分别放置四个无线传感节点，移动机器人安装了**AP**。机器人遵循三种不同的轨迹移动：对角线轨迹、内部轨迹以及边界轨迹。在**simulation**文件中，每种定位算法都有一个单独的**Python**脚本文件。

要运行此脚本，请执行``$ python <script_file>``操作。
## 数据集1（真实世界的公共数据集）
包含与模拟**simulation**文件相同的脚本文件，以便于在**csv**文件可用的数据上运行。
[https://github.com/pspachos/RSSI-Dataset-for-Indoor-Localization-Fingerprinting](https://github.com/pspachos/RSSI-Dataset-for-Indoor-Localization-Fingerprinting)显示了数据集的详细信息。

要运行此文件脚本，请执行``$ python <script_file> <dataset file>``操作。
## 数据集2（真实世界的公共数据集）
包含与模拟**simulation**文件相同的脚本文件，以便于在**csv**文件可用的数据上运行。
[https://ieee-dataport.org/documents/multi-channel-ble-rssi-measurements-indoor-localization](https://ieee-dataport.org/documents/multi-channel-ble-rssi-measurements-indoor-localization)显示了数据集的详细信息。

要运行此文件脚本，请执行``$ python <script_file> <dataset file>``操作。
## 图像
**image**文件展示了仿真实验中，基于集成学习算法对三种机器人轨迹进行预测的结果，文件还包含了数据集**1**中**RSSI**样本点的位置、和数据集**2**中**RSSI**样本点的位置。
