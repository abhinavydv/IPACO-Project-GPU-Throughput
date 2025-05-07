import numpy as np
import matplotlib.pyplot as plt

file = open("plot_data.csv")

data = [list(map(float, i.split(","))) for i in file.readlines()]

data = np.array(data)

# print(data[:, 1], data[:, 2]/data[:, 3])
plt.plot(data[:, 0], data[:, 2]/data[:, 1])
plt.title("Speed-up after replacing 1 FP32 with 1 FP64")
plt.xlabel("Number of operations (n)")
plt.ylabel("Time(n FP32) / TIME(1 FP64 + (n-1) FP32)")
plt.savefig("figs/fp32_fp64_perf2.png")
plt.show()
