import numpy as np
import matplotlib.pyplot as plt

#PROBLEM 1A

A_test = 1.
Nsat=100
a=2.4
b=0.25
c=1.6

def n(x,A,Nsat,a,b,c): #first use the function with A=1
    return A*Nsat*((x/b)**(a-3))*np.exp(-(x/b)**c)

def n_integrand(x):
    return 4*np.pi*(x**2)*n(x,A_test,Nsat,a,b,c)

def trap_loweropen(f,a,b,N): #eval. at semi open interval (a,b]
    x_values = np.linspace(a,b,N+1)[1:]
    y_values = f(x_values)
    h = (b-a)/N #step size 
    return 0.5*h*(y_values[-1]+2*np.sum(y_values[0:N]))

def simpson_loweropen(f,a,b,N): #eval. at semi open interval (a,b]
    S0 = trap_loweropen(f,a,b,N)
    S1 = trap_loweropen(f,a,b,2*N)
    return (4*S1 - S0)/3

n_noNorm = simpson_loweropen(n_integrand,0,5,10000)#use A=1 to compute integral
A = 100/n_noNorm #use Nsat = 100 to find the correct A value 

print('The normalisation value A is:',np.around(A,16))

def romberg_loweropen(f,a,b,m):
    h = (b-a) #stepsize
    r = np.zeros(m)
    r[0] = trap_loweropen(f,a,b,1)
    N_p = 1
    for i in range(1,m-1):
        r[i] = 0
        delta = h
        h = 0.5*h
        x = a+h
        for j in range(N_p):
            r[i] = r[i] + f(x)
            x = x + delta
        r[i] = 0.5*(r[i-1]+delta*r[i])
        N_p = 2*N_p
    N_p = 1 
    for i in range(1,m-1):
        N_p = 4*N_p
        for j in range(0,m-i):
            r[j] = (N_p * r[j+1] - r[j])/(N_p - 1)
    return r[0]

n_noNorm_2 = romberg_loweropen(n_integrand,0,5,20)
print(100/n_noNorm_2)

def n_normalised(x):
    return n(x,(100/n_noNorm),100,2.4,0.25,1.6)

#PROBLEM 1B

def lcg(I):
    a = 31238243
    c = 146778
    m = 2**29
    I = (a*I+c)%m
    return I 

def xor_shift64(I,a1,a2,a3):
    x = lcg(I)
    x = x ^ (x >> a1)
    x = x ^ (x << a2)
    x = x ^ (x >> a3)
    return x 

def RNG(a1,a2,a3,seed,N): #random number generator using combi of lcg and xor
    seed = int(seed)
    random_numbers = np.zeros(N)
    for i in range(N):
        random_numbers[i] = xor_shift64(seed,a1,a2,a3)/(2**64 - 1)
        seed = xor_shift64(seed,a1,a2,a3)
    return random_numbers

seeds = RNG(21,35,4,123456789,int(1e6))

def sample_rejection(f,a,b,seed_list,N):
    sample,f_sample = [], []
    x_values = np.linspace(a,b,N)
    y_values = f(x_values)
    f_max = max(y_values)
    print('max value is ' ,f_max)
    middle_index = int(len(seed_list)/2)
    for i in range(len(seed_list)):
        x = RNG(21,35,4,int(seed_list[i]*10e10),1)*(b-a) + a 
        y = RNG(21,35,4,int(seed_list[middle_index]*10e10),1)*(f_max-a) + a 
        if y < f(x):
            sample.append(x)
            f_sample.append(y)
        middle_index = middle_index + 1 
        if len(sample) == N:
            sample,f_sample = np.array(np.squeeze([sample])),\
                np.array(np.squeeze([f_sample]))
            return sample,f_sample

def distribution(x):
    return 4*np.pi*(x**2)*n_normalised(x)*100

radii = np.linspace(1e-10,5,10000)
p_x = distribution(radii)
logbins = np.logspace(-4,np.log10(5),21)

bin_width = np.zeros(len(logbins)-1)

for i in range(1,len(logbins)):
    bin_width[i-1] = logbins[i]-logbins[i-1]
    
print(bin_width)
print(logbins[1:]/bin_width)

print(logbins[1]-logbins[0])
print(logbins[15]-logbins[14])

coords_sampled = sample_rejection(distribution,1e-10,5,seeds,10000)
radii_sampled = coords_sampled[0]
dist_sampled = coords_sampled[1]

counts = np.histogram(radii_sampled, bins=logbins)[0]
print(counts)
plt.hist(counts,bins=20,density=True,edgecolor='black')
#plt.xscale('log')
plt.show()

#plt.hist(radii_sample_2[0],bins=logbins,density=True,edgecolor='black')
#plt.xscale('log')
#plt.show()

plt.scatter(radii_sampled,dist_sampled,s=0.1,color='black',zorder=10)
plt.hist(radii_sampled,bins=logbins,density=False,histtype='step')
plt.plot(radii,p_x)
plt.xlabel('x')
plt.ylabel('p(x)')
plt.xscale('log')
plt.yscale('log')
plt.show()

plt.scatter(radii_sampled,dist_sampled,s=0.1,color='black',zorder=10)
plt.hist(radii_sampled,bins=logbins,density=False,histtype='step')
plt.plot(radii,p_x)
plt.xlabel('x')
plt.ylabel('p(x)')
plt.xscale('log')
plt.yscale('log')
plt.xlim([1e-4,5])
plt.ylim([1e-30,1e4])
plt.show()

#PROBLEM 1C

"""
random_index = (RNG(21,35,4,5324582,100)*len(sample)).astype(int)
print(random_index)
random_index = random_index.tolist()
print(random_index[10])
print(random_index[0:10]+random_index[10+1:])

def selection(sample,N):
    random_indices = (RNG(21,35,4,5324582,N)*len(sample)).astype(int)
    random_indices = random_indices.tolist()
    for i in range(len(random_indices)):
        arr_without_i = random_indices[0:i]+random_indices[i+1:]
        if random_indices[i] in arr_without_i:
            random_indices[i] = (RNG(21,35,4,267355,1)*len(sample)).tolist()
    random_indices = np.array(random_indices).astype(int)
    return random_indices 

A_test = np.array([1,3,5,2,6,1,7,2,4])
print(selection(A_test,5))
"""

#PROBLEM 1D 

def central_diff(f,x,h):
    return (f(x+h) - f(x-h)) / (2*h)

def n_deriv_analytical(x,A,Nsat,a,b,c):
    return A*Nsat*(np.exp(-(x/b)**c))*((a-3)/b * (x/b)**(a-4) -\
                                       c/b * (x/b)**(a-3) * (x/b)**(c-1))
        
def ridders(f,x,h,d,m): #function, x_values, h, d, order m
    approximations = np.zeros(m)
    approximations[0] = central_diff(f,x,h)
    for i in range(1,m):
        h = h/d
        approximations[i] = central_diff(f,x,h)
    #return approximations
    #print(approximations)
    for i in range(1,m):
        d_power = d**(2*(i+1))
        for j in range(0,m-i):
            approximations[j] = (d_power*approximations[j+1] -\
                                 approximations[j]) / (d_power-1)
    return approximations[0]

import timeit

begin_CD = timeit.default_timer()
derivative_CD = central_diff(n_normalised,1,1e-4) #gives great value 
end_CD = timeit.default_timer() - begin_CD
print(f'it took {end_CD} seconds')
print(np.around(derivative_CD,12))

begin_Ridder = timeit.default_timer()
derivative_Ridder = ridders(n_normalised,1,1e-4,4,5) #gives great value 
end_Ridder = timeit.default_timer() - begin_Ridder
print(f'it took {end_Ridder} seconds')
print(np.around(derivative_Ridder,12))
 
derivative_analytical = n_deriv_analytical(1,(100/n_noNorm),100,2.4,0.25,1.6)
print(np.around(derivative_analytical,12))

"""
x_values = np.linspace(0.9,1.1,1000)
dx = 0.2/1000
y_values = n_normalised(x_values)
derivative = np.diff(y_values)/dx

plt.plot(x_values,y_values)
plt.plot(x_values[1:],derivative)
plt.hlines(-0.6253288977,0.9,1.1,linestyle='dashed')
plt.grid()
plt.show()
"""































