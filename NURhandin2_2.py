import numpy as np

k=1.38e-16 # erg/K
aB = 2e-13 # cm^3 / s

#PROBLEM 2A

# here no need for nH nor ne as they cancel out
def equilibrium1(T,Z,Tc,psi):
    return psi*Tc*k - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T*k


def equilibrium2(T,Z,Tc,psi, nH, A, xi):
    return (psi*Tc - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T - .54 * \
            ( T/1e4 )**.37 * T)*k*nH*aB + A*xi + 8.9e-26 * (T/1e4)
        
def bisection(f,a,b,error): #bisection algorithm to find root
    c_old = 0 
    c_new = (a+b)/2 #set midpoint between start a and end b 
    iteration = 1 #counts the iterations
    while np.abs(c_new-c_old) > error: #repeat as long as |c i+1 - c i| > err
        c_old = c_new
        if f(a)*f(c_old) < 0: #check whether the linear line crosses on left...
            b = c_old #shift boundary b to left 
        elif f(b)*f(c_old) < 0: #or on the right! 
            a = c_old #shift boundary a to right 
        c_new = (a+b)/2 #update c and go again  
        iteration += 1
    return c_new, iteration #return both the root and number of iter
        
def false_position(f,a,b,error): #function f, start a, stop b, acceptable error 
    c_old = 0
    c_new = b + (b-a)*f(b)/(f(a)-f(b)) #use secant formula instead 
    iteration = 1
    while np.abs(c_new-c_old) > error: #and combine the right with bisection
        c_old = c_new
        if f(a)*f(c_old) < 0:
            b = c_old
        elif f(b)*f(c_old) < 0:
            a = c_old
        c_new = b + (b-a)*f(b)/(f(a)-f(b)) #update c and go again 
        iteration += 1
    return c_new, iteration #return both the root and number of iter

def equilibrium1_input(T): #root-finding assumes only one input parameter, x=T
    return equilibrium1(T,0.015,1e4,0.929)

def central_diff(f,x,h): #simple central difference implementation 
    return (f(x+h) - f(x-h)) / (2*h)

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

def newton_raphson(f,x0,error): #function f, estimate x0, acceptable error 
    x0_old = x0
    x0_new = x0_old - f(x0_old)/ridders(f,x0_old,0.0001,4,5) #NR-formula 
    iteration = 1
    while np.abs(x0_new - x0_old) > error: #keep updating x0 until false 
        x0_old = x0_new
        x0_new = x0_old - f(x0_old)/ridders(f,x0_old,0.0001,4,5)
        iteration += 1
    return x0_new, iteration #return both the root and number of iter

def FPNR_combi(f,a,b,error,switch): #attempt at combining FP (1st) and NR (2nd)
    c_old = 0
    c_new = (a*f(b) - b*f(a))/(f(b)-f(a)) #FP formula for c
    iteration = 1
    while np.abs(c_old/c_new) < switch: #switch new parameter: when to go to NR
        c_old = c_new
        if f(a)*f(c_old) < 0:
            b = c_old
        elif f(b)*f(c_old) < 0:
            a = c_old
        c_new = (a*f(b) - b*f(a))/(f(b)-f(a))
        iteration += 1
        
    h = c_new-c_old #use current difference as estimate for h 
    x0_old = c_new
    x0_new = x0_old - f(x0_old)/central_diff(f,x0_old,h)
    while np.abs(x0_new - x0_old) > error: #continue until acceptable error 
        x0_old = np.abs(x0_new)
        x0_new = x0_old - f(x0_old)/central_diff(f,x0_old,h)
        iteration += 1
    return x0_new, iteration #return both the root and number of iter

import timeit #use timeit module to time how fast algorithm runs 

begin_2a = timeit.default_timer()
for i in range(100): #100 steps are more than sufficient to get an average
    T_2a = newton_raphson(equilibrium1_input,3e3,0.1)[0] #logaritmic middle! 
T_2a_iter = newton_raphson(equilibrium1_input,3e3,0.1)[1]
end_2a = (timeit.default_timer()-begin_2a)/100 #get the average time 
print('2a: For the gas:')
print(f'Using {T_2a_iter} iterations, the Newton-Raphson algorithm')
print(f'took {end_2a} seconds')

"""
Commented this as its used for testing the methods but not in final version

begin = timeit.default_timer()
for i in range(10):
    T_2a = false_position(equilibrium1_input,1,1e7,0.1)
end = timeit.default_timer()-begin
print('The time taken in seconds is',end)
begin = timeit.default_timer()
for i in range(10):
    T_2a = bisection(equilibrium1_input,1,1e7,0.1)
end = timeit.default_timer()-begin
print('The time taken in seconds is',end)
"""

#PROBLEM 2B

def equilibrium2_inputCase1(T): #three input cases for low,mid, high density
    return equilibrium2(T,0.015,1e4,0.929,1e-4,5e-10,1e-15)

def equilibrium2_inputCase2(T):
    return equilibrium2(T,0.015,1e4,0.929,1,5e-10,1e-15)

def equilibrium2_inputCase3(T):
    return equilibrium2(T,0.015,1e4,0.929,1e4,5e-10,1e-15)

#important: NR converges for n=1e-4, so we can't use it

"""
Commented this as its used for testing the methods but not in final version

begin = timeit.default_timer()
T_2b_case1 = FPNR_combi(equilibrium2_inputCase1,1,1e15,1e-10,0.77)#least iter!
end = timeit.default_timer()-begin
print('The time taken in seconds is',end)

begin = timeit.default_timer()
T_2b_case1 = false_position(equilibrium2_inputCase1,1,1e15,1e-10)
end = timeit.default_timer()-begin
print('The time taken in seconds is',end)
"""

begin_2b1 = timeit.default_timer() #do equivalent time measure for 2b 
for i in range(10):
    T_2b_case1 = bisection(equilibrium2_inputCase1,1,1e15,1e-10)[0]
T_2b1_iter = bisection(equilibrium2_inputCase1,1,1e15,1e-10)[1]
end_2b1 = (timeit.default_timer()-begin_2b1)/100
print('2b: For the low density gas:')
print(f'Using {T_2b1_iter} iterations, the bisection algorithm')
print(f'took {end_2b1} seconds')

begin_2b2 = timeit.default_timer()
for i in range(100):
    T_2b_case2 = newton_raphson(equilibrium2_inputCase2,3e7,1e-10)[0]
T_2b2_iter = newton_raphson(equilibrium2_inputCase2,3e7,1e-10)[1]
end_2b2 = (timeit.default_timer()-begin_2b2)/100
print('2b: For the intermediate density gas:')
print(f'Using {T_2b2_iter} iterations, the Newton-Raphson algorithm')
print(f'took {end_2b2} seconds')

begin_2b3 = timeit.default_timer()
for i in range(100):
    T_2b_case3 = newton_raphson(equilibrium2_inputCase3,3e7,1e-10)[0]
T_2b3_iter = newton_raphson(equilibrium2_inputCase3,3e7,1e-10)[1]
end_2b3 = (timeit.default_timer()-begin_2b3)/100
print('2b: For the high density gas:')
print(f'Using {T_2b3_iter} iterations, the Newton-Raphson algorithm')
print(f'took {end_2b3} seconds')

output = [[T_2a,T_2a_iter,end_2a],\
          [T_2b_case1,T_2b1_iter,end_2b1],\
          [T_2b_case2,T_2b2_iter,end_2b2],\
          [T_2b_case3,T_2b3_iter,end_2b3]] #writes output to .txt file
    
np.savetxt('NURhandin2problem2.txt',output,fmt='%f')








