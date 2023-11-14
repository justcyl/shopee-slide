import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
# Z = np.maximum(X,Y)
Z=((X/0.84)**6+(Y/0.84)**6) /(2*(1/0.84)**6)

plt.imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap='coolwarm',alpha=0.6)
plt.colorbar()
plt.show()
