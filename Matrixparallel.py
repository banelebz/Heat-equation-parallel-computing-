from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#################################################################################
#function
def prd_u(A,E,O):
 B=np.zeros((lenX,lenY)) 
#I reshape the matrix into a vector
 A=A.reshape(1,lenX*lenX)
# Then i divide that vector into sub matrix depending on how big the original matrix was.
 comm.Scatter(A,E,root=0)
# the dot product
 K=np.dot(E,O)
# i gather them back into a matrix
 comm.Gather(K,B,root=0) 
 return B
#################################################################################
#  starting time
start=time.time()
#################################################################################
lenX = lenY = 20
####################################################################################
# The number of processors must divide the rows
si=int(lenX/size)
########################################################################################
if rank ==0:
 if lenX % size !=0:
  print(" the number of processors must be able to divide the length size, the number you inserted doesn't ")
  comm.Abort()
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
 # Set boundary condition
 T[(lenY-1):,:] = Ttop
 T[:1,:] =Tbottom
 T[:,(lenX-1):] = Tright
 T[:,:1] = Tleft 
 #Boundary conditions for the insulated edges, top and bottom edges 
 for n in range(1,lenY-1):
  T[lenX-1,n]=0.25*(2*T[lenX-2,n] + T[lenX-1,n-1] + T[lenX-1,n+1])
  T[0,n]=0.25*(2*T[1,n] + T[0,n-1] + T[0,n+1])
 Tu=T 
else:
 Tu = np.zeros((lenX,lenY))
 T = np.zeros((lenX,lenY))
###################################################################################
# broadcasing to other ranks
comm.Bcast(Tu,root=0)
comm.Bcast(T,root=0)
t=0
Bu = np.zeros((si,lenY))
###################################################################################
#Matrix of the teperature of the wire
h=1/lenX
f=np.zeros(lenX*lenY).reshape(lenX,lenY) 
k=1.1234e-4               
f[int(lenX/2),int(lenY/2)]=(1500)*(h**2/k)
convergence=1
#######################################################################################
#creating sparse matrix
n=lenX
main = np.ones(n)
offset_one = np.ones(n-1)
offset_one1 = np.ones(n-2)
T1=np.diag(offset_one, k = -1) + np.diag(offset_one, k = 1)
######################################################################################
# Matrix to help with the bountry conditions of insulated region
T3=np.zeros((lenX,lenY))
T3[lenX-2,lenY-1]=T3[lenX-2,lenY-1] +1
T3[1,0]= T3[1,0] +1
Tu=T
#######################################################################################
#computating
while convergence>1e-6:  
 Ti2=prd_u(T1,Bu,T)
 Ti3=prd_u(T,Bu,T1)
 Ti4=prd_u(T,Bu,T3)
 T=0.25*(Ti2 + Ti3 + Ti4 + f)
 comm.Bcast(T,root=0)
 Tu=np.array(T)
#######################################################################################
#convergence
 if t==0:
  convergence=1
 else:
  convergence=(abs((Tk-Tu).max()))/abs(Tu.max())       #Computing convergence
 Tk=Tu  
 t=t+1
####################################################################################
#
end=time.time()
duration=end-start
######################################################################################
if rank==0:
 print ("wait a second")
 print ("Iteration finished" )
 np.savetxt('output11.txt',T,fmt='%d')
 #configure the contour
 #plt.title(" contour of Temperature")
 #plt.contourf(X,Y,T,colorinterpolation,cmap=colourMap)
 #Confiration the colorbar
 #plt.colorbar()
 #Show the result in the plot window 
 #plt.show()
 print(duration)
 print(t)



