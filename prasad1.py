import numpy as np
import matplotlib.pyplot as plt
a = np.array([0, 0, 0, 0 ,0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],dtype='float32')
b = np.array([-1,1], dtype='float32')
c = np.convolve(a,b)
plt.subplot(211)
plt.plot(a, 'o-')
plt.subplot(212)
plt.plot(c, 'o-')
plt.show()

