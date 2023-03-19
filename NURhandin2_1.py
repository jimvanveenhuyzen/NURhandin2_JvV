import numpy as np
import matplotlib.pyplot as plt

#PROBLEM 1A

A = 1.
Nsat=100
a=2.4
b=0.25
c=1.6

def n(x,A,Nsat,a,b,c):
    return A*Nsat*((x/b)**(a-3))*np.exp(-(x/b)**c)

def n_integrand(x):
    return n(x,A,Nsat,a,b,c)

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

print('The normalisation value A is:',np.around(A,6))

#PROBLEM 1B

































