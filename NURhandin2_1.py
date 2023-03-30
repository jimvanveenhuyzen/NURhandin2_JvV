import numpy as np
import matplotlib.pyplot as plt

#PROBLEM 1A

A_test = 1. #parameters from the problem 
Nsat=100
a=2.4
b=0.25
c=1.6

def n(x,A,Nsat,a,b,c): #first use the function with A=1
    return A*Nsat*((x/b)**(a-3))*np.exp(-(x/b)**c)

def n_integrand(x): #fill in parameters so that n(x) only a function of x 
    return 4*np.pi*(x**2)*n(x,A_test,Nsat,a,b,c) #useful for e.g. integration

def trap_loweropen(f,a,b,N): #eval. at semi open interval (a,b]
    x_values = np.linspace(a,b,N+1)[1:] #normally we wouldn't exclude 0th elem
    y_values = f(x_values)
    h = (b-a)/N #step size chosen 
    return 0.5*h*(y_values[-1]+2*np.sum(y_values[0:N])) #formula from slides

def romberg_loweropen(f,a,b,m): #input: function f, start a, stop b, order m
    h = (b-a) #stepsize
    r = np.zeros(m) #array of initial guesses
    r[0] = trap_loweropen(f,a,b,1000) #using N=1000 for initial guess  
    N_p = 1
    for i in range(1,m-1): #range from 1 to m-1 
        r[i] = 0 #set the other initial estimates to 0 
        delta = h
        h = 0.5*h #reduce stepsize 
        x = a+h
        for j in range(N_p): #0 to N_p 
            r[i] = r[i] + f(x)
            x = x + delta 
        r[i] = 0.5*(r[i-1]+delta*r[i]) #new estimate of r[i]
        N_p = 2*N_p #increase N_p as i increases
    N_p = 1 #reset N_p to 1
    for i in range(1,m-1):
        N_p = 4*N_p #increase N_p (very fast) for increasing i
        for j in range(0,m-i):
            r[j] = (N_p * r[j+1] - r[j])/(N_p - 1) #new estimate 
    return r[0] #final value: all estimates merged into a final result at r[0]

A_true = Nsat/romberg_loweropen(n_integrand,0,5,20) #compute the true A value 
print(f'The normalisation constant is {A_true}')


def n_normalised(x): #same as n(x), but now only as a func of x and with true A
    return n(x,A_true,100,2.4,0.25,1.6)

#PROBLEM 1B

def lcg(I): #linear congruential generator, using some arbitrary a,c and m. 
    a = 31238243 
    c = 146778
    m = 2**29 #has to be an order of 2, close to 2**32 
    I = (a*I+c)%m
    return I 

def xor_shift64(I,a1,a2,a3): #XORSHIFT 64 bit random number 
    x = lcg(I) #take result of LCG
    x = x ^ (x >> a1)
    x = x ^ (x << a2)
    x = x ^ (x >> a3) 
    return x #returns a value after XOR shifting it around 3 times 

def RNG(a1,a2,a3,seed,N): #random number generator using combi of lcg and xor
    seed = int(seed) #set as integer to use as an input for XOR
    random_numbers = np.zeros(N) #to fill an array of length N with RNs
    for i in range(N):
        random_numbers[i] = xor_shift64(seed,a1,a2,a3)/(2**64 - 1)
        seed = xor_shift64(seed,a1,a2,a3) #create new usable seed for next iter
    return random_numbers #random number array of size N 

seeds = RNG(21,35,4,123456789,int(1e6)) #generate 1 million random seeds to use

def sample_rejection(f,a,b,seed_list,N): #func, U(a,b), seed list, length N
    sample,f_sample = [], [] #arrays to fill with N (x,f(x)) values 
    x_values = np.linspace(a,b,N)
    y_values = f(x_values)
    f_max = max(y_values) #use this value f_max for y in Uniform(a,f_max) dist! 
    middle_index = int(len(seed_list)/2) #arbitrary
    for i in range(len(seed_list)):
        x = RNG(21,35,4,int(seed_list[i]*10e10),1)*(b-a) + a  #Uniform(a,b)
        y = RNG(21,35,4,int(seed_list[middle_index]*10e10),1)*(f_max-a) + a 
        if y < f(x): #rejection condition, if rejected, won't append to sample
            sample.append(x)
            f_sample.append(y)
        middle_index = middle_index + 1 
        if len(sample) == N: #the desired length N is reached
            sample,f_sample = np.array(np.squeeze([sample])),\
                np.array(np.squeeze([f_sample]))
            return sample,f_sample #returns (x,f(x)) coordinates of sample

def distribution(x): #p(x) = N(x)/100
    return 4*np.pi*(x**2)*n_normalised(x)/Nsat

radii = np.linspace(1e-4,5,10000) #so we can plot p(x) for the x-range
p_x = distribution(radii) 
logbins = np.logspace(-4,np.log10(5),20) #create bins as instructed

coords_sampled = sample_rejection(distribution,1e-4,5,seeds,10000)#10000 galaxy
radii_sampled = coords_sampled[0] #x coordinate of galaxy
dist_sampled = coords_sampled[1] #p(x) of galaxy

plt.scatter(radii_sampled,dist_sampled*Nsat,s=0.1,\
            color='black',zorder=10,label='sampled points')#(x,N(x)) of sample
plt.hist(radii_sampled,bins=logbins,density=True,edgecolor='black',\
         label='histogram') #density=True so we divide by bin width as asked!
plt.plot(radii,p_x*Nsat,label='N(x)') #plot p(x) to compare the plots properly
plt.xlabel('x')
plt.ylabel('N(x) = p(x)$<N_{sat}>$')
plt.xscale('log')
plt.yscale('log')
plt.xlim([1e-4,5])
plt.ylim([1e-30,1e4])
plt.title('log-log plot showing N(x) and hist of sampled points')
plt.legend()
plt.savefig("./problem1b.png")
plt.show()
plt.close()

#PROBLEM 1C

def select(sample,seed_list,N): #make a selection of N galaxies from a sample 
    selected_sample = np.array([])
    index = int(len(seed_list)/4) #arbitrary, but must use N_seed >> N_sample
    for i in range(len(seed_list)): #random index from U(0,10.000) galaxies
        random_index = int(RNG(21,35,4,int(index*10e10),1)*len(sample))
        while np.any(selected_sample == sample[random_index]): 
            #this condition checks we don't draw same galaxy twice!
            random_index = int(RNG(21,35,4,int(index*10e10),1)*len(sample))
        selected_sample = np.append(selected_sample, sample[random_index])
        if len(selected_sample) == N: #desired sample size reached 
            return selected_sample #we now have N data points from the sample 
        index = index + 1 
        
def selection_sort(array): #very simple selection sorting algorithm from slides
    N = len(array)
    for i in range(N-1): 
        i_min = i #take the current i as i_min, update if a[i_min] not lowest!
        for j in range(i+1,N):
            if array[j] < array[i_min]:
                i_min = j #update i_min if the element at j is smaller
        if i_min != i:
            array[i_min],array[i] = array[i],array[i_min] #swap the elements!
    return array #returns sorted array 

radii_selected = select(radii_sampled,seeds,100) #select sample of 100 galaxies
radii_selected_sorted = selection_sort(radii_selected) #sort radii of galaxies

def n_normalised_integrand(x): #seperate function for N(x) for convenience 
    return 4*np.pi*(x**2)*n_normalised(x)

Nsat_selected = np.zeros(len(radii_selected_sorted)) #using integration, we
for i in range(len(radii_selected_sorted)): #find N(<x) for all 100 galaxies 
    Nsat_selected[i] = romberg_loweropen(n_normalised_integrand\
                                         ,0,radii_selected_sorted[i],5)
        
plt.plot(radii_selected_sorted,Nsat_selected,color='black') #xlog plot of N(<x)
plt.xscale('log')
plt.xlim([1e-4,5])
plt.xlabel('x')
plt.ylabel('N(<x)')
plt.title('Number of galaxies $N(<x)$ within radius $x$')
plt.savefig("./problem1c.png")
plt.show()
plt.close()

#PROBLEM 1D 

def central_diff(f,x,h): #simple central difference implementation 
    return (f(x+h) - f(x-h)) / (2*h)

def n_deriv_analytical(x,A,Nsat,a,b,c): #expression for analytical derivative 
    return A*Nsat*(np.exp(-(x/b)**c))*((a-3)/b * (x/b)**(a-4) -\
                                       c/b * (x/b)**(a-3) * (x/b)**(c-1))
        
def ridders(f,x,h,d,m): #function f, x value, free params: h, d, order m
    approximations = np.zeros(m) #order m determines number of approximations
    approximations[0] = central_diff(f,x,h) #use central diff for initial apprx
    for i in range(1,m): 
        h = h/d
        approximations[i] = central_diff(f,x,h) #compute aprx for decreasing h
    for i in range(1,m): #next we use neville-like algorithm to combine approx
        d_power = d**(2*(i+1))
        for j in range(0,m-i):
            approximations[j] = (d_power*approximations[j+1] -\
                                 approximations[j]) / (d_power-1)
    return approximations[0] #first element contains combined approximations


derivative_Ridder = ridders(n_normalised,1,1e-4,4,5)#call ridders to find f'(x)
print('Using Ridders method, dn/dx at x=1 is',np.around(derivative_Ridder,15))
 
derivative_analytical = n_deriv_analytical(1,A_true,100,2.4,0.25,1.6)
print('The analytic value of dn/dx at x=1 is',\
      np.around(derivative_analytical,15)) #simply fill into analytical expres.
    
output = [A_true,derivative_Ridder,derivative_analytical] #output to .txt file

np.savetxt('NURhandin2problem1.txt',output,fmt='%1.12f')





























