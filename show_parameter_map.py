# from result.txt file, make 2d function ploat
# 1st column: parameter1, 2nd column: parameter2, 3rd column: result

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# read data from result.txt
data = np.loadtxt('result.txt')

# make 2d function plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2])
ax.set_xlabel('parameter1')
ax.set_ylabel('parameter2')
ax.set_zlabel('result')
plt.show()

# make it as 2d colormap plot with meshgrid
x = np.linspace(min(data[:,0]), max(data[:,0]), 100)
y = np.linspace(min(data[:,1]), max(data[:,1]), 100)

z = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        z[j,i] = data[np.argmin((data[:,0]-x[i])**2 + (data[:,1]-y[j])**2),2]

X, Y = np.meshgrid(x, y)
fig, ax = plt.subplots()
c = ax.contourf(X, Y, z, cmap='viridis')
fig.colorbar(c, ax=ax)
plt.xlabel('parameter1')
plt.ylabel('parameter2')
plt.title('Parameter Map')
plt.show()
