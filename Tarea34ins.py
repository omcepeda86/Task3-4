
import random
import math
import matplotlib.pyplot as plt
import time
import numpy as np
from numpy.random import default_rng


def objective_function1(O,D,C):
    
    #### constrain 1###########################
    ### get sums by user doing constrain
    sumac2 = []
    sumaind = 0
    #number constrains
    Rs = len(O)/len(D)
    for id, i in enumerate(O):
        sumaind = sumaind + i
        if ((Rs-1) == id%Rs):
            sumac2.append(sumaind)
            sumaind = 0

    ## unment constrain 1
    sumapen = 0
    for jd, j in enumerate(D):
        if(sumac2[jd] <= j):
            sumapen = sumapen + 10000
    ###########################################
    
    
    
    ### objective function
    SumF1 = 0
    for id, i in enumerate(O):
        SumF1 = SumF1 + (10 + C[id]*i)
    
    ## complete objective function
    z = SumF1 + sumapen

    return z


### crreate instance

## sets
#####################################################3
#### colocar acá la cantidad de dispositivos, usuarios y periodos de tiempo
us = ['a','b']#,'c'] # user
dis = [1,2] # device
tn = [1,2]#,3] # period of time
###############################################################


### parameter cost
Ci = [np.random.randint(10) for i in us for j in dis for t in tn]
#PCi = {(i,j,t): ((len(dis))*(len(us))*id1)+((len(us))*td + jd) for id1,i in enumerate(us) for jd, j in enumerate(dis) for td, t in enumerate(tn)}
print('los costos para cada variable son:')
print(Ci)
#print(PCi)
##############################################################3

nv =  len(Ci) # number of variables
mm = - 1  # if minimization problem, mm = -1; if maximization problem, mm = 1
  
bounds = [(0, 50) for i in range(nv)] 

################################################################3
Dem = [np.random.randint(10) for i in us for j in dis]
#PDem = {(i,j): Dem[((len(us)-1)*id1 + jd)] for id1,i in enumerate(us) for jd, j in enumerate(dis)}
#Dem[ (jd*(len(us)-1)) + id1]
print('la demanda de cada usuario en cada dispositivo es:')
print(Dem)
#print(PDem)
###############################################################3#



# PARAMETERS OF PSO
particle_size = 500  # number of particles
iterations = 100  # max number of iterations
w = 0.8  # inertia constant
c1 = 2  # cognative constant
c2 = 0.5  # social constant
  
# Visualization
fig = plt.figure()
ax = fig.add_subplot()
fig.show()
plt.title('Evolutionary process of the objective function value')
plt.xlabel("Iteration")
plt.ylabel("Objective function")
# ------------------------------------------------------------------------------
class Particle:
    def __init__(self, bounds):
        self.particle_position = []  # particle position
        self.particle_velocity = []  # particle velocity
        self.local_best_particle_position = []  # best position of the particle
        self.fitness_local_best_particle_position=initial_fitness
        self.fitness_particle_position=initial_fitness

        for i in range(nv):
            self.particle_position.append(
                random.uniform(bounds[i][0],bounds[i][1]))
            self.particle_velocity.append(random.uniform(-1,1))
            
    def evaluate(self, objective_function1):
        self.fitness_particle_position = objective_function1(self.particle_position, Dem, Ci)
        if mm == -1:
            if self.fitness_particle_position < self.fitness_local_best_particle_position:
                self.local_best_particle_position = self.particle_position  # update the local best
                self.fitness_local_best_particle_position = self.fitness_particle_position  # update the fitness of the local best
        if mm==1:
            if self.fitness_particle_position > self.fitness_local_best_particle_position:
                self.local_best_particle_position = self.particle_position  # update the local best
                self.fitness_local_best_particle_position = self.fitness_particle_position  # 


  
    def update_velocity(self, global_best_particle_position):
        for i in range(nv):
            r1 = random.random()
            r2 = random.random()
  
            cognitive_velocity = c1 * r1 * (self.local_best_particle_position[i] - self.particle_position[i])
            social_velocity = c2 * r2 * (global_best_particle_position[i] - self.particle_position[i])
            self.particle_velocity[i] = w * self.particle_velocity[i] + cognitive_velocity + social_velocity
  
    def update_position(self, bounds):
        for i in range(nv):
            self.particle_position[i] = self.particle_position[i] + self.particle_velocity[i]
  
            # check and repair to satisfy the upper bounds
            if self.particle_position[i] > bounds[i][1]:
                self.particle_position[i] = bounds[i][1]
            # check and repair to satisfy the lower bounds
            if self.particle_position[i] < bounds[i][0]:
                self.particle_position[i] = bounds[i][0]
  
class PSO:
    def __init__(self, objective_function1, bounds, particle_size, iterations):
        fitness_global_best_particle_position = initial_fitness
        global_best_particle_position = []
        swarm_particle = []
        for i in range(particle_size):
            swarm_particle.append(Particle(bounds))
        A=[]
          
        for i in range(iterations):
            for j in range(particle_size):
                swarm_particle[j].evaluate(objective_function1)        
                
                if mm==-1:
                    if swarm_particle[j].fitness_particle_position < fitness_global_best_particle_position:
                        global_best_particle_position = list(swarm_particle[j].particle_position)
                        fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)

                if mm==1:
                    if swarm_particle[j].fitness_particle_position>fitness_global_best_particle_position:
                        global_best_particle_position = list(swarm_particle[j].particle_position)
                        fitness_global_best_particle_position=float(swarm_particle[j].fitness_particle_position)
        
            for j in range(particle_size):
                swarm_particle[j].update_velocity(global_best_particle_position)
                swarm_particle[j].update_position(bounds)
                
            A.append(fitness_global_best_particle_position)
            
        print("Result:")
        print("Optimal solution:",global_best_particle_position)
        print("Objective function",fitness_global_best_particle_position)
        ax.plot(A,color="r")
        fig.canvas.draw()
        ax.set_xlim(left=max(0,i-iterations),right=i+3)
        time.sleep(0.01)
        
        ### impresión solución
        #Xf = {(i,j,t): global_best_particle_position[td + jd + id1]  for id1,i in enumerate(us) for jd, j in enumerate(dis) for td, t in enumerate(tn)}
        #print(Xf)

if mm==-1:
    initial_fitness=float("inf")
if mm==1:
    initial_fitness=-float("inf")


PSO(objective_function1,bounds,particle_size,iterations)



## parameter
## demand of user i en dispositive j




plt.show()









