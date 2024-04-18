# File: huber.py
# Author: Juan F. Osorio
# Date: May 2024
# Math for DS 

# Import libraries
import jax
from jax import vmap, grad, random, jit
import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.style.use("ggplot")


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
epsilon = sgm*random.normal(shape = (m-5,), key = key1)
epsilon_outliers = 100*sgm*random.normal(shape = (5,), key = key1)
epsilon = jnp.concatenate([epsilon,epsilon_outliers])
# Generate y
y = jnp.dot(A,xopt) + epsilon


# Huber function
def huber(w, eta):
    res = jnp.where(jnp.abs(w)>eta, eta*(jnp.abs(w) - 0.5*eta), 0.5*w**2)
    return res

print(huber(jnp.array([12,3,4]),3.))

# Cost function
eta = jnp.std(y)
def L(A,x,y,lmbd):
    # Data miss fit
    #data_missfit = jnp.sum([huber(jnp.dot(A[i,:],x) - y(i), eta) for i in range(m)])
    data_missfit = jnp.sum(huber(jnp.dot(A,x) - y, eta))
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
for i in range(100):
    x = x - lr*grad_L(A,x,y,lmbd)
    # Loss at current iterate
    loss = L(A,x,y,lmbd)
    loss_GD.append(loss)
    print('Iteration {}, Loss {}'.format(i,loss))

# Accelarated gradient descent

# Learning rate
lr = 1/beta
# Initial value for x and a's
x = x0
xprev = x0
a = 1
aprev = 1
# Store values of the loss
loss_AGD = []
for i in range(100):
    u = x + a*(1/aprev - 1)*(x - xprev)
    xnext = u - 1/beta*grad_L(A,u,y,lmbd)
    # Loss at current iterate
    loss = L(A,xnext,y,lmbd)
    loss_AGD.append(loss)
    print('Iteration {}, Loss {}'.format(i,loss))
    # Update x's
    xprev = jnp.copy(x)
    x = jnp.copy(xnext)
    # Set extrapolation coefficient
    aprev = jnp.copy(a)
    a = 0.5*(jnp.sqrt(a**4 + 4*a**2) - a**2)


# Plot
plt.figure(figsize=(9, 7))
plt.semilogy(loss_GD, )
plt.semilogy(loss_AGD)
plt.title('Huber-Ridge regression: GD vs AGD')
plt.legend(["GD", "AGD"], loc="upper right")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig('huberout.png',dpi=300)

