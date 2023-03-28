import numpy as np
import matplotlib.pyplot as plt

k=1.38e-16 # erg/K
aB = 2e-13 # cm^3 / s


# here no need for nH nor ne as they cancel out
def equilibrium1(T,Z,Tc,psi):
    return psi*Tc*k - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T*k


def equilibrium2(T,Z,Tc,psi, nH, A, xi):
    return (psi*Tc - (0.684 - 0.0416 * np.log(T/(1e4 * Z*Z)))*T - .54 * \
            ( T/1e4 )**.37 * T)*k*nH*aB + A*xi + 8.9e-26 * (T/1e4)
        
def false_position(f,a,b,error):
    c_old = 0
    c_new = b + (b-a)*f(b)/(f(a)-f(b))
    iteration = 1
    print(f'After {iteration} iterations the root is estimated at {c_new}')
    while np.abs(c_new-c_old) > error:
        c_old = c_new
        if f(a)*f(c_old) < 0:
            b = c_old
        elif f(b)*f(c_old) < 0:
            a = c_old
        c_new = b + (b-a)*f(b)/(f(a)-f(b))
        iteration += 1
        print(f'After {iteration} iterations the root is estimated at {c_new}')
    return c_new

def equilibrium1_input(T):
    return equilibrium1(T,0.015,1e4,0.929)

def central_diff(f,x,h):
    return (f(x+h) - f(x-h)) / (2*h)

def newton_raphson(f,x0,error):
    x0_old = x0
    x0_new = x0_old - f(x0_old)/central_diff(f,x0_old,0.0001)
    iteration = 1
    print(f'After {iteration} iterations root is estimated at ({x0_new},{f(x0_new)})')
    while np.abs(x0_new - x0_old) > error:
        x0_old = x0_new
        x0_new = x0_old - f(x0_old)/central_diff(f,x0_old,0.0001)
        iteration += 1
        print(f'After {iteration} iterations root is estimated at ({x0_new},{f(x0_new)})')
    return x0_new

print(newton_raphson(equilibrium1_input,3e3,0.1))

def equilibrium2_inputCase1(T):
    return equilibrium2(T,0.015,1e4,0.929,1e-4,5e-10,1e-15)

def equilibrium2_inputCase2(T):
    return equilibrium2(T,0.015,1e4,0.929,1,5e-10,1e-15)

def equilibrium2_inputCase3(T):
    return equilibrium2(T,0.015,1e4,0.929,1e4,5e-10,1e-15)

#print(false_position(equilibrium2_inputCase2,1,1e15,1e-10))

print(false_position(equilibrium2_inputCase1,1,1e15,1e-10))

#important: NR converges for n=1e-4, so we can't use it
#print(newton_raphson(equilibrium2_inputCase1,3e7,1e-10))
#print(newton_raphson(equilibrium2_inputCase2,3e7,1e-10))
#print(newton_raphson(equilibrium2_inputCase3,3e7,1e-10))

def FPNR_combi(f,a,b,error):
    c_old = 0
    c_new = (a*f(b) - b*f(a))/(f(b)-f(a))
    iteration = 1
    print(f'After {iteration} iterations the root is estimated at {c_new}')
    while np.abs(c_old/c_new) < 0.99:
        c_old = c_new
        if f(a)*f(c_old) < 0:
            b = c_old
        elif f(b)*f(c_old) < 0:
            a = c_old
        c_new = (a*f(b) - b*f(a))/(f(b)-f(a))
        iteration += 1
        print(f'After {iteration} iterations the root is estimated at {c_new}')
        
    h = c_new-c_old
    x0_old = c_new
    x0_new = x0_old - f(x0_old)/central_diff(f,x0_old,h)
    print(f'After {iteration} iterations root is estimated at ({x0_new},{f(x0_new)})')
    while np.abs(x0_new - x0_old) > error:
        h = x0_new-x0_old
        x0_old = np.abs(x0_new)
        x0_new = x0_old - f(x0_old)/central_diff(f,x0_old,h)
        iteration += 1
        print(f'After {iteration} iterations root is estimated at ({x0_new},{f(x0_new)})')
    return x0_new

#print(FPNR_combi(equilibrium2_inputCase1,1,1e15,1e-10))
#print(newton_raphson(equilibrium2_inputCase2,3e7,1e-10))
#print(newton_raphson(equilibrium2_inputCase3,3e7,1e-10))

#print(FPNR_combi(equilibrium1_input,1,1e7,0.1))












