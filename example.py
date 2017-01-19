import numpy as np
import matplotlib.pyplot as plt
from RBFN import RBFN

x = np.linspace(0,10,100)
y = np.sin(x)
model = RBFN(input_shape = 1, hidden_shape = 10)
model.fit(x,y)
y_pred = model.predict(x)

plt.plot(x,y,'b-',label='real')
plt.plot(x,y_pred,'r-',label='fit')
plt.legend(loc='upper right')
plt.title('Interpolation using a RBFN')
plt.show()
