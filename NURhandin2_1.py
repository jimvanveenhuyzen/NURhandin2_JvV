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

def romberg_loweropen(f,a,b,m):
    h = (b-a) #stepsize
    r = np.zeros(m)
    r[0] = trap_loweropen(f,a,b,1000)
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

A_true = Nsat/romberg_loweropen(n_integrand,0,5,20)
print(f'The normalisation constant is {A_true}')


def n_normalised(x):
    return n(x,A_true,100,2.4,0.25,1.6)

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

seeds = RNG(21,35,4,123456789,int(1e6)) #generate 1 million random seeds to use

def sample_rejection(f,a,b,seed_list,N):
    sample,f_sample = [], []
    x_values = np.linspace(a,b,N)
    y_values = f(x_values)
    f_max = max(y_values)
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
    return 4*np.pi*(x**2)*n_normalised(x)/Nsat

radii = np.linspace(1e-4,5,10000)
p_x = distribution(radii)
logbins = np.logspace(-4,np.log10(5),20)

coords_sampled = sample_rejection(distribution,1e-4,5,seeds,10000)
radii_sampled = coords_sampled[0]
dist_sampled = coords_sampled[1]

plt.scatter(radii_sampled,dist_sampled*Nsat,s=0.1,\
            color='black',zorder=10,label='sampled points')
plt.hist(radii_sampled,bins=logbins,density=True,edgecolor='black',\
         label='histogram')
plt.plot(radii,p_x*Nsat,label='N(x)')
plt.xlabel('x')
plt.ylabel('N(x) = p(x)$<N_{sat}>$')
plt.xscale('log')
plt.yscale('log')
plt.xlim([1e-4,5])
plt.ylim([1e-30,1e4])
plt.title('log-log plot showing N(x) and hist of sampled points')
plt.legend()
plt.show()

#PROBLEM 1C

def select(sample,seed_list,N):
    selected_sample = np.array([])
    index = int(len(seed_list)/4)
    for i in range(len(seed_list)):
        random_index = int(RNG(21,35,4,int(index*10e10),1)*len(sample))
        while np.any(selected_sample == sample[random_index]): 
            #this condition checks we don't draw same galaxy twice!
            random_index = int(RNG(21,35,4,int(index*10e10),1)*len(sample))
        selected_sample = np.append(selected_sample, sample[random_index])
        if len(selected_sample) == N:
            return selected_sample
        index = index + 1 
        
def selection_sort(array):
    N = len(array)
    for i in range(N-1):
        i_min = i
        for j in range(i+1,N):
            if array[j] < array[i_min]:
                i_min = j
        if i_min != i:
            array[i_min],array[i] = array[i],array[i_min]
    return array

radii_selected = select(radii_sampled,seeds,100)
radii_selected_sorted = selection_sort(radii_selected)

def n_normalised_integrand(x):
    return 4*np.pi*(x**2)*n_normalised(x)

Nsat_selected = np.zeros(len(radii_selected_sorted))
for i in range(len(radii_selected_sorted)):
    Nsat_selected[i] = romberg_loweropen(n_normalised_integrand\
                                         ,0,radii_selected_sorted[i],5)
        
plt.plot(radii_selected_sorted,Nsat_selected,color='black')
plt.xscale('log')
plt.xlim([1e-4,5])
plt.xlabel('x')
plt.ylabel('N(<x)')
plt.title('Number of galaxies $N(<x)$ within radius $x$')
plt.show()

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


derivative_Ridder = ridders(n_normalised,1,1e-4,4,5) #gives great value 
print('Using Ridders method, dn/dx at x=1 is',np.around(derivative_Ridder,15))
 
derivative_analytical = n_deriv_analytical(1,A_true,100,2.4,0.25,1.6)
print('The analytic value of dn/dx at x=1 is',\
      np.around(derivative_analytical,15))
    
derivatives = [derivative_Ridder,derivative_analytical]

np.savetxt('NURhandin2_problem1a.txt', [A_true])
np.savetxt('NURhandin2_problem1d.txt', derivatives)





























