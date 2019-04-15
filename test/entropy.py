import sympy as sym

# x = sym.symbols('x')
# sym.plotting.plot(-sym.log(x), (x, 0, 1))
# sym.plotting.plot(-x * sym.log(x), (x, 0, 1))
# sym.plotting.plot(-x * sym.log(x) - (1 - x) * sym.log(1 - x), (x, 0, 1))
import numpy as np
import matplotlib.pyplot as plt

# print()
x = np.linspace(.001, 1 - .001, 10000)
# y = -x * np.log2(x) - (1 - x) * np.log2(1 - x)
# plt.scatter(x, y)
# plt.xlabel('x')
# plt.ylabel('H(x)')
# plt.show()
y = -np.log2(x)
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('info(x)')
plt.show()


