#!/usr/bin/env python
# coding: utf-8

# # MATH 210 Assignment 5
# 
# ### Instructions
# 
# * There are 4 problems and 22 total points.
# * Write your solutions in the cells below.
# * You may work on these problems with others but you must write your solutions on your own.
# * Use NumPy, SciPy and Matplotlib.
# * Execute the test cells to verify that your solutions pass.
# * **Grading includes hidden tests!** Your solution may not be completely correct even if it passes all tests below.
# * Submit this notebook to Canvas before **11:59pm Monday November 25**

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.integrate as spi


# ### Problem 1 (7 points)
# 
# Write a function called `is_adjacency` which takes an input parameter `A`, a 2D NumPy array, and returns `True` if $A$ is the adjacency matrix of simple, connected, undirected graph and `False` otherwise. In other words, the function should return `True` if all the following conditions are satisfied:
# * $A$ is square
# * $A$ is symmetric
# * All diagonal entries of $A$ are 0
# * All off-diagonal entries of $A$ are either 0 or 1 (this implies that the graph associated with $A$ is simple and undirected)
# * The multiplicity of the eigenvalue 0 in the Laplacian matrix $L$ associated with $A$ is 1 (this implies that the graph associated with $A$ is connected)
# 
# Note that `la.eig` returns complex numbers with real and imaginary parts as floats. It's not good practice to use `a == 0` to test if a float is equal to zero since functions like `la.eig` may have small errors. Instead use something like `np.abs(a) < 10e-15` to test if a float is small enough that we consider it equal to zero.
# 
# There are several NumPy functions that may be helpful such as [`np.diag`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.diag.html), [`np.all`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.all.html), [`np.allclose`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html), and [`np.unique`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html). Use any NumPy functions that you like.
# 
# The [Laplacian matrix](https://en.wikipedia.org/wiki/Laplacian_matrix) $L$ of a graph $G$ is defined as the matrix where the entry at index $(i,j)$ is -1 if node $i$ is connected to node $j$, 0 if not and $-d_i$ at the diagonal entry at index $(i,i)$ where $d_i$ is the degree of node $i$. The [degree](https://en.wikipedia.org/wiki/Degree_(graph_theory)) of a node is the number of edges connected to that node. The degree of node $i$ is the sum of row $i$ in the adjacency matrix $A$. In other words, $L = D - A$ where $A$ is the adjacency matrix and $D$ is the [degree matrix](https://en.wikipedia.org/wiki/Degree_matrix).
# 
# For example, let $G$ be the following graph:
# 
# ![graph](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/6n-graf.svg/440px-6n-graf.svg.png)
# 
# The adjacency matrix is
# 
# $$
# A_G = \begin{bmatrix}
# 0 & 1 & 0 & 0 & 1 & 0 \\
# 1 & 0 & 1 & 0 & 1 & 0 \\
# 0 & 1 & 0 & 1 & 0 & 0 \\
# 0 & 0 & 1 & 0 & 1 & 1 \\
# 1 & 1 & 0 & 1 & 0 & 0 \\
# 0 & 0 & 0 & 1 & 0 & 0
# \end{bmatrix}
# $$
# 
# The degree matrix is
# 
# $$
# D_G = \begin{bmatrix}
# 2 & 0 & 0 & 0 & 0 & 0 \\
# 0 & 3 & 0 & 0 & 0 & 0 \\
# 0 & 0 & 2 & 0 & 0 & 0 \\
# 0 & 0 & 0 & 3 & 0 & 0 \\
# 0 & 0 & 0 & 0 & 3 & 0 \\
# 0 & 0 & 0 & 0 & 0 & 1
# \end{bmatrix}
# $$
# 
# Note the diagonal entries are the sum of the rows of $A_G$. The Laplacian matrix is:
# 
# $$
# L_G = D_G - A_G = \left[ \begin{array}{rrrrrr}
# 2 & -1 & 0 & 0 & -1 & 0 \\
# -1 & 3 & -1 & 0 & -1 & 0 \\
# 0 & -1 & 2 & -1 & 0 & 0 \\
# 0 & 0 & -1 & 3 & -1 & -1 \\
# -1 & -1 & 0 & -1 & 3 & 0 \\
# 0 & 0 & 0 & -1 & 0 & 1
# \end{array} \right]
# $$
# 
# Finally the multiplicity of the eigenvalue 0 in $L_G$ is 1 as we see when we compute:

# In[111]:


L = np.array([[2,-1,0,0,-1,0],
              [-1,3,-1,0,-1,0],
              [0,-1,2,-1,0,0],
              [0,0,-1,3,-1,-1],
              [-1,-1,0,-1,3,0],
              [0,0,0,-1,0,1]])

eigvals,eigvecs = la.eig(L)
eigvals = np.real(eigvals)
print(np.round(eigvals,2))


# In[136]:


def is_adjacency(A):
    if np.shape(A)[1] != np.shape(A)[0]:
        return False
    elif not np.allclose(A,A.T):
        return False
    elif not np.allclose(np.diag(A),np.zeros(np.shape(A)[1])):
        return False
    elif not (len(np.unique(A)) == 2 and np.unique(A)[0] == 0 and np.unique(A)[1] == 1):
        return False
    D = np.identity(np.shape(A)[1])
    for n in range(0,np.shape(A)[1]):
        D[n,n] = sum(A[n])
    L = D - A
    eigvals,eigvecs = la.eig(L)
    eigvals = np.round(np.real(eigvals),5)
    nums,counts = np.unique(eigvals,return_counts = True)
    if not (nums[0] == 0 and counts[0] == 1):
        return False
    return True


# In[138]:


"Check that is_adjacency returns the correct type."
A = np.array([[0,1],[1,0]])
assert type(is_adjacency(A)) == bool , "Return value should be boolean."
print("Problem 1 Test 1: Success!")


# In[139]:


"Check that is_adjacency returns False when A is not square."
A = np.arange(0,21).reshape(3,7)
assert not is_adjacency(A) , "Return value should be False."
print("Problem 1 Test 2: Success!")


# In[140]:


"Check that is_adjacency returns False when A is not symmetric."
A = np.arange(0,25).reshape(5,5)
assert not is_adjacency(A) , "Return value should be False."
print("Problem 1 Test 3: Success!")


# In[141]:


"Check that is_adjacency returns False when A has non-zero diagonal entries."
M = np.arange(0,16).reshape(4,4)
A = M + M.T
assert not is_adjacency(A) , "Return value should be False."
print("Problem 1 Test 4: Success!")


# In[142]:


"Check that is_adjacency returns False when A has entries not equal to 0 or 1."
M = np.arange(0,16).reshape(4,4)
B = M + M.T
A = B - np.diag(np.diag(B))
assert not is_adjacency(A) , "Return value should be False."
print("Problem 1 Test 5: Success!")


# In[143]:


"Check that is_adjacency returns False when A has Laplacian with eigenvalue 0 with multiplicity bigger than 1."
A = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])
assert not is_adjacency(A) , "Return value should be False."
print("Problem 1 Test 6: Success!")


# In[144]:


"Check that is_adjacency returns True when A is the adjacency matrix."
A = np.array([[0,1,0,1],[1,0,0,0],[0,0,0,1],[1,0,1,0]])
assert is_adjacency(A) , "Return value should be True."
print("Problem 1 Test 7: Success!")


# ##  Problem 2 (5 points)
# 
# Write a function called `trig_euler` which takes input parameters `y0`, `tf` and `N` (with default values `tf=10` and `N=100`) and implements Euler's method with step size $h = \frac{t_f}{N-1}$ to approximate the solution of the equation
# 
# $$
# y' = \sin(y^2) + \cos(t) \ , \ y(0) = y_0
# $$
# 
# over the interval $[0,t_f]$. The function plots the approxmiation $y(t)$ over the interval $[0,t_f]$ and returns a 2D NumPy array `Y` of size $N$ by 2 where the column at index 0 is the array of $t$ values from 0 to $t_f$, and the column at index 1 is the array of corresponding $y$ values given by Euler's method.

# In[182]:


def trig_euler(y0,tf = 10,N = 100):
    h = tf / (N-1)
    y = np.zeros(N)
    y[0] = y0
    t = np.linspace(0,tf,N)
    for n in range(0,N-1):
        y[n+1] = y[n] +  (np.sin(y[n]**2) + np.cos(t[n]))*h
    return np.stack((t,y),axis = -1)


# In[183]:


"Check that trig_euler returns an array of correct size."
N = 500
assert np.allclose(trig_euler(-1,20,N=N).shape,[N,2]), "Return type should be a NumPy array of size N by 2."
print("Problem 2 Test 1: Success!")


# In[184]:


"Check that trig_euler returns t values."
result = trig_euler(3,tf=5,N=101)
assert np.allclose(result[:5,0],np.array([0.,0.05,0.1,0.15,0.2]))
print("Problem 2 Test 2: Success!")


# In[185]:


"Check that trig_euler returns y values."
y1 = np.array([0,5])
result1 = trig_euler(0,tf=5,N=2)
assert np.allclose(result1[:,1],y1), "Return values should be y=5 at t=5 in this case."
y2 = np.array([-1.,-0.76981613,-0.57597472,-0.41414877,-0.27650039])
result2 = trig_euler(-1,N=25,tf=3)
assert np.allclose(result2[:5,1],y2)
print("Problem 2 Test 3: Success!")


# ##  Problem 3 (5 points)
# 
# Write a function called `oscillator` which takes input parameters `m`, `c`, `k`, `f` (a Python function of one variable), `y0`, `v0`, `tf` and `N` (with default values `tf=10` and `N=100`) and implements Euler's method with step size $h = \frac{t_f}{N-1}$ to approximate the solution of the equation
# 
# $$
# my'' + cy' + ky = f(t) \ , \ \ y(0)=y_0 \ , \ y'(0) = v_0
# $$
# 
# over the interval $[0,t_f]$. The function plots the approxmiation $y(t)$ over the interval $[0,t_f]$ and returns a 2D NumPy array `Y` of size $N$ by 2 where the column at index 0 is the array of $t$ values from 0 to $t_f$, and the column at index 1 is the array of corresponding $y$ values given by Euler's method.

# In[186]:


def oscillator(m,c,k,f,y0,v0,tf=10,N=100):
    h = tf / (N-1)
    y = np.zeros(N)
    v = np.zeros(N)
    y[0] = y0
    v[0] = v0 
    t = np.linspace(0,tf,N)
    for n in range(0,N-1):
        y[n+1] = y[n] + h*v[n]
        v[n+1] = v[n] + h*((f(t[n]) - k*y[n]-c*v[n])/m)
    return np.stack((t,y),axis = -1) 


# In[187]:


"Check that oscillator returns an array of correct size."
N = 200
result = oscillator(2,1,3,np.cos,0,2,50,N=N)
assert np.allclose(result.shape,[N,2]), "Return type should be a NumPy array of size N by 2."
print("Problem 3 Test 1: Success!")


# In[188]:


"Check that oscillator returns t values."
result = oscillator(1,0,1,lambda t: 0,1,0,1,N=11)
assert np.allclose(result[:5,0],np.array([0.,0.1,0.2,0.3,0.4]))
print("Problem 3 Test 2: Success!")


# In[189]:


"Check that trig_euler returns y values."
y1 = np.array([1.,1.,0.75])
result1 = oscillator(1,0,1,lambda t: 0,1,0,1,N=3)
assert np.allclose(result1[:,1],y1)
y2 = np.array([0.,0.39215686,0.7266436,0.9892295,1.17109905])
result2 = oscillator(4,3,7,lambda t: np.sin(2*t),0,2,10,N=52)
assert np.allclose(result2[:5,1],y2)
print("Problem 3 Test 3: Success!")


# ##  Problem 4 (5 points)
# 
# Write a function called `coriolis` which takes input parameters `w`, `Omega`, `phi`, `u0` and `t`. The function uses `scipy.integrate.odeint` to numerically solve the system of equations
# 
# \begin{align*}
# \frac{d^2 x}{dt^2} &= -\omega^2 x + 2 \Omega \sin(\phi) \frac{dy}{dt} \\
# \frac{d^2 y}{dt^2} &= -\omega^2 y - 2 \Omega \sin(\phi) \frac{dx}{dt}
# \end{align*}
# 
# given the initial conditions $[x(t_0),x'(t_0),y(t_0),y'(t_0)]$ defined by `u0` (a Python list of length 4), and where $t$ is a 1D NumPy array of $t$ values over an interval $[t_0,t_f]$. The parameters $\omega$, $\Omega$, and $\phi$ are given by `w`, `Omega`, and `phi` respectively. The function plots the trajectory $(x(t),y(t))$ of the solution and returns a 2D NumPy array $M$ of size `len(t)` by 3 where the column at index 0 is the array of $t$ values, the column at index 1 is the corresponding array of $x$ values and the column at index 2 is the corresponding array of $y$ values.

# In[32]:


def f(u,t,w,Omega,phi):
    dudt = np.array([0.0,0.0,0.0,0.0])
    dudt[0] = u[1]
    dudt[1] = -w**2*u[0] + 2*Omega*np.sin(phi)*u[3]
    dudt[2] = u[3]
    dudt[3] = -w**2*u[2] - 2*Omega*np.sin(phi)*u[1]
    return dudt

def coriolis(w,Omega,phi,u0,t):
    u = spi.odeint(f,u0,t,args=(w,Omega,phi))
    return np.stack((t,u[:,0],u[:,2]),axis=-1)


# In[35]:


"Check that coriolis returns an array of the correct size."
assert np.allclose(coriolis(3,3,np.pi/6,[1,0,1,0],np.linspace(0,20,200)).shape, [200,3])
assert np.allclose(coriolis(2,1,np.pi/2,[1,-1,1,2],np.linspace(0,10,100)).shape, [100,3])
print("Problem 5 Test 1: Success!")


# In[36]:


"Check that coriolis returns the correct values."
solution = coriolis(2,1,np.pi/2,[1,-1,1,2],np.linspace(0,10,200))
M = np.array([[ 0.        ,  1.        ,  1.        ],
              [ 0.05025126,  0.94974454,  1.09781044],
              [ 0.10050251,  0.89943111,  1.18959012],
              [ 0.15075377,  0.84891533,  1.2744251 ]])
assert np.allclose(solution[:4,:],M)

print("Problem 5 Test 3: Success!")


# In[37]:


"Check that coriolis returns the correct values."
solution = coriolis(2,1,np.pi/2,[1,-1,1,2],np.linspace(0,10,200))

M = np.array([[9.89949749,1.306117,0.6589853],
              [9.94974874,1.29575448,0.76147343],
              [10.,1.2822863,0.85746863]])
assert np.allclose(solution[-3:,:],M)

print("Problem 5 Test 3: Success!")

