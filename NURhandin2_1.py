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

def trapezoid(f,a,b,N):
    """
    Parameters
    ----------
    f : function
        Takes in a function y=f(x) to perform integration over.
    a : float or int 
        Lower boundary of the integral.
    b : float or int 
        Upper boundary of the integral.
    N : int
        Number of points used, higher N increases accuracy but is more costly

    Returns
    -------
    float
        Returns value of the integral.
    """
    x_values = np.linspace(a,b,N+1)
    y_values = f(x_values)
    h = (b-a)/N #step size 
    return 0.5*h*(y_values[0]+y_values[-1]+2*np.sum(y_values[1:N]))

def simpson(f,a,b,N):
    S0 = trapezoid(f,a,b,N)
    S1 = trapezoid(f,a,b,2*N)
    return (4*S1 - S0)/3

A_values = np.linspace(9.1,9.3,10000)

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

radii_sample = RNG(21,35,4,123456789,10000)*5  #Unif(0,5) distribution
#radii_sample = np.random.rand(10000)*5
print(radii_sample)

plt.hist(radii_sample,bins=20,density=True,edgecolor='black')
plt.xlabel('x')
plt.show()

def n_normalised_integrand(x):
    return 4*np.pi*(x**2)*n_normalised(x)/100

radii = np.linspace(1e-10,5,10000)
p_x = n_normalised_integrand(radii)

sampled_points = np.zeros(10000)
for i in range(10000):
    sampled_points[i] = n_normalised_integrand(radii_sample[i])
    
def rejection(f,x):
    y = np.array([])
    for i in range(len(x)):
        if x[i] <= f(x[i]):
            y = np.append(y,x[i])
    return y 
    
sample = rejection(n_normalised_integrand,radii_sample)
print(len(radii_sample),len(sample))
print(max(sample))
sample_y = n_normalised_integrand(sample)
logbins = np.logspace(-4,np.log10(5),20)
    
plt.scatter(sample,sample_y,s=1,color='black',zorder=10)
plt.hist(sample,bins=logbins,density=True,edgecolor='black')
plt.plot(radii,p_x)
plt.xlabel('x')
plt.ylabel('p(x)')
plt.xscale('log')
plt.yscale('log')
plt.show()

plt.scatter(sample,sample_y,s=1,color='black',zorder=10)
plt.hist(sample,bins=logbins,density=True,edgecolor='black')
plt.plot(radii,p_x)
plt.xlabel('x')
plt.ylabel('p(x)')
plt.xscale('log')
plt.yscale('log')
plt.xlim([1e-4,5])
plt.ylim([1e-30,10])
plt.show()

#print(logbins)

plt.hist(sample,bins=logbins,density=False,edgecolor='black')
plt.xscale('log')
plt.show()

#PROBLEM 1D 

def central_diff(f,x,h):
    return (f(x+h) - f(x-h)) / (2*h)

derivative_CD = central_diff(n_normalised,1,1e-6) #gives great value 
print(derivative_CD)

x_values = np.linspace(0.9,1.1,1000)
dx = 0.2/1000
y_values = n_normalised(x_values)
derivative = np.diff(y_values)/dx

plt.plot(x_values,y_values)
plt.plot(x_values[1:],derivative)
plt.hlines(-0.6253288977,0.9,1.1,linestyle='dashed')
plt.grid()
plt.show()































