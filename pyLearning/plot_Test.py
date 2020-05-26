from matplotlib import pyplot as plt
import numpy as np
a = np.array([0,1,1.5,2,3,3.5])
b = 0.5 * a + 1
plt.title("my function")
plt.xlabel(xlabel='x')
plt.ylabel(ylabel='y')
plt.plot(a,b,'ob')
plt.show()