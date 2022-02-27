import numpy as np
import matplotlib.pyplot as plt
# I used this program to draw graph from output from parallel graph
##################################################################################################
# The length used in the parallel program
lenY=lenX=20
###################################################################################################
X,Y = np.meshgrid(np.arange(0,lenX),np.arange(0,lenY))
colorinterpolation =50
colourMap = plt.cm.jet #you can tey: colourMap = plt.cm.coolwarm
#configure the contour

T=np.genfromtxt('output11.txt')
T=T.reshape(lenX,lenY)
#print (T)
plt.title(" contour of Temperature")
plt.contourf(X,Y,T,colorinterpolation,cmap=colourMap)
#Confiration the colorbar
plt.colorbar()
#Show the result in the plot window 
plt.show()

