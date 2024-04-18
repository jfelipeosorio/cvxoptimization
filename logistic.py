# File: logistic.py
# Author: Juan F. Osorio
# Date: May 2024
# Math for DS 

# Import libraries
import numpy as np
import jax
from jax import vmap, grad, random, jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.style.use("ggplot")


# Given parameters in the problem

# Number of observations
m = 100  
# Number of covariates
n = 50
# Regularization strenght
lmbd = 1
# Variance in error
sgm = jnp.sqrt(0.25)
# Beta
beta = 100


# Data generation for problem

# Generate design matrix
key1 = random.key(1)
X = random.normal(shape=(m,n), key = key1)
# Generate theta optimal
thetaopt = jnp.ones((n,))
# Generate gaussian error
epsilon = sgm*random.normal(shape = (m,), key = key1)
# Generate z: True
z = jnp.dot(X,thetaopt)
# Generate p: True probabilities
p = 1/(1 + jnp.exp(-z))
# Generate y from p
y = np.array(jax.random.bernoulli(key = key1, p=p))
y[y == 1] = 1
y[y == 0] = -1
y = jnp.array(y)
# Cost function

def L(X,theta,y,lmbd):
    # Data miss fit
    data_missfit = jnp.sum(jnp.log(1 + jnp.exp(-jnp.multiply(y,jnp.dot(X,theta)))))
    # Regularization
    reg = lmbd/2*jnp.sum(jnp.power(theta,2))
    return data_missfit + reg


# Gradient descent algorithm

# Learning rate
lr = 1/beta
# Initial value of x
key2 = random.key(2)
theta0 = random.normal(shape=(n,), key = key2)
# Create gradient function
grad_L = jax.jit(grad(L, argnums=1))

# Gradient descent

# Initial value for x
theta = theta0
# Store values of the loss
loss_GD = []
# Iterate
for i in range(100):
    theta = theta - lr*grad_L(X,theta,y,lmbd)
    # Loss at current iterate
    loss = L(X,theta,y,lmbd)
    loss_GD.append(loss)
    print('Iteration {}, Loss {}'.format(i,loss))

# Accelarated gradient descent

# Learning rate
lr = 1/beta
# Initial value for theta and a's
theta = theta0
thetaprev = theta0
a = 1
aprev = 1
# Store values of the loss
loss_AGD = []
for i in range(100):
    u = theta + a*(1/aprev - 1)*(theta - thetaprev)
    thetanext = u - 1/beta*grad_L(X,u,y,lmbd)
    # Loss at current iterate
    loss = L(X,thetanext,y,lmbd)
    loss_AGD.append(loss)
    print('Iteration {}, Loss {}'.format(i,loss))
    # Update theta's
    thetaprev = jnp.copy(theta)
    theta = jnp.copy(thetanext)
    # Set extrapolation coefficient
    aprev = jnp.copy(a)
    a = 0.5*(jnp.sqrt(a**4 + 4*a**2) - a**2)


# Plot
plt.figure(figsize=(9, 7))
plt.semilogy(loss_GD, )
plt.semilogy(loss_AGD)
plt.title('Logistic regression: GD vs AGD')
plt.legend(["GD", "AGD"], loc="upper right")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('logisticout.png',dpi=300)

