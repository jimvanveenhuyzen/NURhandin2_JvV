import numpy as np

k=1.38e-16 # erg/K
aB = 2e-13 # cm^3 / s

#PROBLEM 2A
print('PROBLEM 2A')

# here no need for nH nor ne as they cancel out
def equilibrium1(T,Z,Tc,psi):
    return psi*Tc*k - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T*k


def equilibrium2(T,Z,Tc,psi, nH, A, xi):
    return (psi*Tc - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T - .54 * \
            ( T/1e4 )**.37 * T)*k*nH*aB + A*xi + 8.9e-26 * (T/1e4)
        
def bisection(f,a,b,error): #is this not bisection? 
    c_old = 0 
    c_new = (a+b)/2
    iteration = 1
    while np.abs(c_new-c_old) > error:
        c_old = c_new
        if f(a)*f(c_old) < 0:
            b = c_old
        elif f(b)*f(c_old) < 0:
            a = c_old
        c_new = (a+b)/2
        iteration += 1
    return c_new, iteration 
        
def false_position(f,a,b,error):
    c_old = 0
    c_new = b + (b-a)*f(b)/(f(a)-f(b))
    iteration = 1
    while np.abs(c_new-c_old) > error:
        c_old = c_new
        if f(a)*f(c_old) < 0:
            b = c_old
        elif f(b)*f(c_old) < 0:
            a = c_old
        c_new = b + (b-a)*f(b)/(f(a)-f(b))
        iteration += 1
    return c_new, iteration

def equilibrium1_input(T):
    return equilibrium1(T,0.015,1e4,0.929)

def central_diff(f,x,h):
    return (f(x+h) - f(x-h)) / (2*h)

def ridders(f,x,h,d,m): #function, x_values, h, d, order m
    approximations = np.zeros(m)
    approximations[0] = central_diff(f,x,h)
    for i in range(1,m):
        h = h/d
        approximations[i] = central_diff(f,x,h)
    for i in range(1,m):
        d_power = d**(2*(i+1))
        for j in range(0,m-i):
            approximations[j] = (d_power*approximations[j+1] -\
                                 approximations[j]) / (d_power-1)
    return approximations[0]

def newton_raphson(f,x0,error):
    x0_old = x0
    x0_new = x0_old - f(x0_old)/ridders(f,x0_old,0.0001,4,5)
    iteration = 1
    while np.abs(x0_new - x0_old) > error:
        x0_old = x0_new
        x0_new = x0_old - f(x0_old)/ridders(f,x0_old,0.0001,4,5)
        iteration += 1
    return x0_new, iteration

def FPNR_combi(f,a,b,error,switch):
    c_old = 0
    c_new = (a*f(b) - b*f(a))/(f(b)-f(a))
    iteration = 1
    while np.abs(c_old/c_new) < switch:
        c_old = c_new
        if f(a)*f(c_old) < 0:
            b = c_old
        elif f(b)*f(c_old) < 0:
            a = c_old
        c_new = (a*f(b) - b*f(a))/(f(b)-f(a))
        iteration += 1
        
    h = c_new-c_old
    x0_old = c_new
    x0_new = x0_old - f(x0_old)/central_diff(f,x0_old,h)
    while np.abs(x0_new - x0_old) > error:
        h = x0_new-x0_old
        x0_old = np.abs(x0_new)
        x0_new = x0_old - f(x0_old)/central_diff(f,x0_old,h)
        iteration += 1
    return x0_new, iteration

import timeit

begin_2a = timeit.default_timer()
for i in range(100):
    T_2a = newton_raphson(equilibrium1_input,3e3,0.1)[0] #logaritmic middle
T_2a_iter = newton_raphson(equilibrium1_input,3e3,0.1)[1]
end_2a = (timeit.default_timer()-begin_2a)/100
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
print('PROBLEM 2B')

def equilibrium2_inputCase1(T):
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

begin_2b1 = timeit.default_timer()
for i in range(10):
    T_2b_case1 = bisection(equilibrium2_inputCase1,1,1e15,1e-10)[0]
T_2b1_iter = bisection(equilibrium2_inputCase1,1,1e15,1e-10)[1]
end_2b1 = (timeit.default_timer()-begin_2b1)/100
print(f'Using {T_2b1_iter} iterations, the bisection algorithm')
print(f'took {end_2b1} seconds')

begin_2b2 = timeit.default_timer()
for i in range(100):
    T_2b_case2 = newton_raphson(equilibrium2_inputCase2,3e7,1e-10)[0]
T_2b2_iter = newton_raphson(equilibrium2_inputCase2,3e7,1e-10)[1]
end_2b2 = (timeit.default_timer()-begin_2b2)/100
print(f'Using {T_2b2_iter} iterations, the Newton-Raphson algorithm')
print(f'took {end_2b2} seconds')

begin_2b3 = timeit.default_timer()
for i in range(100):
    T_2b_case3 = newton_raphson(equilibrium2_inputCase3,3e7,1e-10)[0]
T_2b3_iter = newton_raphson(equilibrium2_inputCase3,3e7,1e-10)[1]
end_2b3 = (timeit.default_timer()-begin_2b3)/100
print(f'Using {T_2b3_iter} iterations, the Newton-Raphson algorithm')
print(f'took {end_2b3} seconds')

output = [[T_2a_iter,end_2a],\
          [T_2b1_iter,end_2b1],\
          [T_2b2_iter,end_2b2],\
          [T_2b3_iter,end_2b3]]
    
np.savetxt('NURhandin2problem2.txt',output,fmt='%f')








