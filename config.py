import random

random.seed(1)  # 固定随机数种子

sensorset_1 = [[50, 25], [62, 30], [50, 35], [62, 40], [50, 45], [62, 50], [50, 55], [62, 60],
               [50, 65], [62, 70], [50, 75], [62, 80], [50, 85], [62, 90], [50, 95], [62, 100],
               [50, 105], [62, 110], [50, 115], [62, 120], [50, 125], [62, 130], [50, 135], [62, 140],
               [50, 145], [62, 150], [50, 155], [62, 160], [50, 165], [62, 170], [50, 175], [62, 180],
               [72, 25], [40, 30], [72, 35], [40, 40], [72, 45], [40, 50], [72, 55], [40, 60],
               [72, 65], [40, 70], [72, 75], [40, 80], [72, 85], [40, 90], [72, 95], [40, 100],
               [72, 105], [40, 110], [72, 115], [40, 120], [72, 125], [40, 130], [72, 135], [40, 140],
               [72, 145], [40, 150], [72, 155], [40, 160], [72, 165], [40, 170], [72, 175], [40, 180]]  # 0.2%, 64个

sensorset_2 = random.sample(sensorset_1, 32)  # 0.1%, 32个

sensorset_3 = random.sample(sensorset_1, 62)  # 缺失2个传感器

sensorset = [sensorset_1, sensorset_2, sensorset_3]