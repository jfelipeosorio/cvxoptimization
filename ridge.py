# File: ridge.py
# Author: Juan F. Osorio
# Date: May 2024
# Math for DS 

# Import libraries
import jax
from jax import vmap, grad, random, jit
import jax.numpy as jnp
import matplotlib as plt


# Given parameters in the problem

# Number of observations
m = 80  
# Number of covariates
n = 100
# Regularization strenght
lmbd = 1
# Variance in error
sgm = jnp.sqrt(0.25)
# Beta
beta = 4*(m+n) + lmbd


# Data generation for problem

# Generate design matrix
key1 = random.key(1)
A = random.normal(shape=(m,n), key = key1)
# Generate x optimal
xopt = random.normal(shape=(n,), key = key1)
# Generate gaussian error
epsilon = sgm*random.normal(shape = (m,), key = key1)
# Generate y
y = jnp.dot(A,xopt) + epsilon


# Cost function

def L(A,x,y,lmbd):
    # Data miss fit
    data_missfit = 0.5*jnp.sum(jnp.power(jnp.dot(A,x)-y,2))
    reg = lmbd/2*jnp.sum(jnp.power(x,2))
    return data_missfit + reg


# Gradient descent algorithm

# Learning rate
lr = 1/beta
# Initial value of x
key2 = random.key(2)
#x0 = random.normal(shape=(n,), key = key2)
x0 = xopt
# Create gradient function
grad_L = jax.jit(grad(L, argnums=1))

# Gradient descent

# Initial value for x
x = x0
# Store values of the loss
loss_GD = []
# Iterate
for i in range(51):
    x = x - lr*grad_L(A,x,y,lmbd)
    # Loss at current iterate
    loss = L(A,x,y,lmbd)
    loss_GD.append(loss)
    print('Iteration {}, Loss {}'.format(i,loss))

plt.plot(loss_GD)
plt.show()

