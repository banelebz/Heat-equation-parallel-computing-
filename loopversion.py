import numpy as np
import matplotlib.pyplot as plt
import time
####################################################################################
start=time.time()
####################################################################################
k=1.1234e-4
#set Dimensin and Delta
lenX = lenY = 30
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
#Boundary conditions for the insulated edges, top and bottom edges
for n in range(1,lenY-1):
 T[n,lenX-1]=0.25*(2*T[lenX-2,n] + T[lenX-1,n-1] + T[lenX-1,n+1])
 T[n,0]=0.25*(2*T[1,n] + T[0,n-1] + T[0,n+1])
#T[int(lenX/2),int(lenY/2)]=T[int(lenX/2),int(lenY/2)]+ f
#Iteration ( We assume that the iteration is convergence in MaxTteration = 500
print ("wait a second")
#for iteration in range(0,maxIter):
convergence=1
t=0
Tu=T
while convergence>1e-6:
 for i in range(1,lenX-1):
  for j in range(1,lenY-1):
   Tu[i,j]=0.25*(Tu[i+1][j]+Tu[i-1][j] + Tu[i][j+1] + Tu[i][j-1] + f[i][j])
 for n in range(1,lenY-1): #Computing the new values of temperature for the insulated edges
  Tu[lenX-1,n]=0.25*(2*Tu[lenX-2,n] + Tu[lenX-1,n-1] + Tu[lenX-1,n+1])
  Tu[0,n]=0.25*(2*Tu[1,n] + Tu[0,n-1] + Tu[0,n+1])
 T2=np.array(Tu)
 if t==0:
  convergence=1
 else:
  convergence=(abs((T-T2).max()))/abs(T2.max())       #Computing convergence
 T=T2
 t=t+1
print ("Iteration finished" )
########################################################################################
end=time.time()
duration = end-start
########################################################################################
#configure the contour
plt.title(" contour of Temperature")
plt.contourf(X,Y,T,colorinterpolation,cmap=colourMap)
#Confiration the colorbar
plt.colorbar()
#Show the result in the plot window 
plt.show()
print(duration)
#print (T)

