import numpy as np
import matplotlib.pyplot as plt
import time

########################################################################################
start= time.time()
########################################################################################
k=1.1234e-4
#set Dimensin and Delta
lenX = lenY = 50
h=1/lenX
# Boundary condition
Ttop=27
Tbottom = 0
Tleft=25
Tright=25
               
# Initial guess of interior grid
Tguess = 25
# set colour interpolation nd colour map
colorinterpolation =50
colourMap = plt.cm.jet #you can tey: colourMap = plt.cm.coolwarm

# set meshgrid
X,Y = np.meshgrid(np.arange(0,lenX),np.arange(0,lenY))
#Set array size and set the interior value with Tguee
T = np.empty((lenX,lenY))
T.fill(Tguess)
f=np.zeros(lenX*lenY).reshape(lenX,lenY)                #Matrix of the teperature of the wire
# Set boundary condition
f[int(lenX/2),int(lenY/2)]=(1500)*(h**2/k)
T[(lenY-1):,:] = Ttop
T[:1,:] =Tbottom
T[:,(lenX-1):] = Tright
T[:,:1] = Tleft
convergence=1
t=0
Tu=T
####################################################################################
#creating the sparse matrix
n=lenX
main = np.ones(n)
offset_one = np.ones(n-1)
offset_one1 = np.ones(n-2)
T1=np.diag(offset_one, k = -1) + np.diag(offset_one, k = 1)
####################################################################################
#creating a matrix to help with insulted bountry conditions
T3=np.zeros((lenX,lenY))
T3[lenX-2,lenY-1]=T3[lenX-2,lenY-1] +1
T3[1,0]= T3[1,0] +1
###################################################################################
# intial insulated region  bountry country conditions
for n in range(1,lenY-1):
 T[n,lenX-1]=0.25*(2*T[lenX-2,n] + T[lenX-1,n-1] + T[lenX-1,n+1])
 T[n,0]=0.25*(2*T[1,n] + T[0,n-1] + T[0,n+1])
t=0 
while convergence>1e-10:
#Boundary conditions for the insulated edges, top and bottom edges
 T= 0.25*(np.dot(T1,T) + np.dot(T,T1) + np.dot(T,T3) + f) 
 Tu=np.array(T)              #Writting the new temperature matrix in numpy array
#
#
 if t==0:
  convergence=1
 else:
  convergence=(abs((Tk-Tu).max()))/abs(Tu.max())       #Computing convergence
 Tk=Tu
 t=t+1
 
print ("wait a second")
print ("Iteration finished" )
####################################################################################
end = time.time()
duration=end-start
####################################################################################
#configure the contour
plt.title(" contour of Temperature")
plt.contourf(X,Y,T,colorinterpolation,cmap=colourMap)
#Confiration the colorbar
plt.colorbar()
#Show the result in the plot window 
plt.show()
print(duration)
print(t)
################################################################################

