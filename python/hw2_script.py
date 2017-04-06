import os
import csv
import numpy as np
import LinReg
import matplotlib.pyplot as plt

[W,X,Y,Z] = LinReg.LinReg_ReadInputs('../Data')

matrix,lossa,lossb,a = LinReg.LinReg_SGD(W,X,Y,Z)

print a

plt.plot(lossa)
plt.plot(lossb)
plt.ylabel('losserror - green test error, blue train error')
plt.xlabel('epochs')
plt.title("LinRegPLOT")
plt.show()
