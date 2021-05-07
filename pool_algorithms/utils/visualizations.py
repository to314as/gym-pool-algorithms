import numpy as np
import matplotlib.pyplot as plt


bottom = 8
max_height = 4
radii = np.load("C:/Users/evolj/Documents/GitHub/Temporary-CS229-pool/src/utils/brute_force_logs.npy",allow_pickle=True)
radii=radii[:,2].reshape(360*2,4)[:,3]+2
print(radii)
N = len(radii)
theta = np.linspace(np.pi, 3 * np.pi, N, endpoint=False)
width = (2*np.pi) / N

ax = plt.subplot(111, polar=True)
bars = ax.bar(theta, radii, width=width, bottom=bottom)

# Use custom colors and opacity
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.jet(r / 10.))
    bar.set_alpha(0.8)

plt.savefig('C:/Users/evolj/Documents/GitHub/Temporary-CS229-pool/demo3.png', transparent=True)
plt.show()

cords=np.load("positions.npy")
print(cords)
plt.scatter(cords[:,0],cords[:,1])
plt.savefig('C:/Users/evolj/Documents/GitHub/Temporary-CS229-pool/reachable_points.png', transparent=True)
plt.show()