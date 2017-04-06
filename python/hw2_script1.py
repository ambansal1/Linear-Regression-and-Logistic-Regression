import os
import csv
import numpy as np
import LogReg
import matplotlib.pyplot as plt
[W,X,Y,Z] = LogReg.LogReg_ReadInputs('../Data')


a,b,c,d,pe   = LogReg.LogReg_SGA(W,X,Y,Z)
e    =   LogReg.LogReg_CalcObj(W, X, a)

print
plt.plot(b)
plt.plot(c)
plt.ylabel('losserror - green test error, blue train error')
plt.xlabel('iterations')
plt.title("LogRegPLOT")
plt.show()


