Lab 1 

1.2
from sympy import *
x= Symbol ('x')
y= Symbol ('y')
z= Symbol ('z')
w3= integrate ( x ** 2+y ** 2 ,y , x )
pprint ( w3 )
w4= integrate ( x ** 2+y ** 2 ,x , y )
pprint ( w4 )


1.3

from sympy import *
x= Symbol ('x')
y= Symbol ('y')
#a= Symbol ('a ')
#b= Symbol ('b ')
a=4
b=6
w3=4* integrate (1 ,( y ,0 ,( b/a )* sqrt ( a ** 2-x ** 2 ) ) ,(x ,0 , a ) )
print ( w3 )

Lab 2 

Ex 2

from sympy import *
x= symbols ('x')
w1= integrate (exp (-x )*x ** 4 ,( x ,0 , float ('inf ') ) )
print ( simplify ( w1 ) )

Ex 5

from sympy import beta , gamma
m= float ( input ('m : ') ) ;
n= float ( input ('n :') ) ;
s= beta (m , n ) ;
t= gamma ( n )
print ('gamma (',n ,') is %3.3f '%t )
print ('Beta (',m ,n ,') is %3.3f '%s )

Ex 6

from sympy import beta, gamma

m = 5
n = 7

m = float(m)
n = float(n)

s = beta(m, n)
t = (gamma(m) * gamma(n)) / gamma(m + n)

print(s, t)

if abs(s - t) <= 0.00001:
    print('beta and gamma are related')
else:
    print('given values are wrong')



Lab 3

1.2

1) 

from sympy . vector import *
from sympy import symbols
N= CoordSys3D ('N') # Setting the coordinate system
x ,y , z= symbols ('x y z')
A=N . x ** 2*N . y+2*N . x*N . z-4 # Variables x,y,z to be used with coordinate
system N
delop =Del () #Del operator
display ( delop ( A ) ) #Del operator applied to A
gradA = gradient ( A ) # Gradient function is used
print ( f"\n Gradient of {A} is \n")
display ( gradA )

2)

from sympy . vector import *
from sympy import symbols
N= CoordSys3D ('N')
x ,y , z= symbols ('x y z')
A=N . x ** 2*N . y*N . z*N . i+N . y ** 2*N . z*N . x*N . j+N . z ** 2*N . x*N . y*N . k
delop =Del ()
divA = delop .dot ( A )
display ( divA )
print ( f"\n Divergence of {A} is \n")
display ( divergence ( A ) )

3) 


from sympy . vector import *
from sympy import symbols
N= CoordSys3D ('N')
x ,y , z= symbols ('x y z')
A=N . x ** 2*N . y*N . z*N . i+N . y ** 2*N . z*N . x*N . j+N . z ** 2*N . x*N . y*N . k
delop =Del ()
curlA = delop . cross ( A )
display ( curlA )
print ( f"\n Curl of {A} is \n")
display ( curl ( A ) )


Lab 4

4.2

import numpy as np
from scipy.linalg import null_space

# Define a linear transformation interms of matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Find the rank of the matrix A
rank = np.linalg.matrix_rank(A)
print("Rank of the matrix ", rank)

# Find the null space of the matrix A
ns = null_space(A)
print("Null space of the matrix ", ns)

# Find the dimension of the null space
nullity = ns.shape[1]
print("Dimension of the null space ", nullity) # Corrected print statement

# Verify the rank-nullity theorem
if rank + nullity == A.shape[1]:
    print("Rank-nullity theorem holds.")
else:
    print("Rank-nullity theorem does not hold.")


4.3


import numpy as np
# Define the vector space V
V = np . array ([
[1 , 2 , 3],
[2 , 3 , 1],
[3 , 1 , 2]])
# Find the dimension and basis of V
basis = np . linalg . matrix_rank ( V )
dimension = V . shape [0]
print (" Basis of the matrix ", basis )
print (" Dimension of the matrix ", dimension )
Lab 5

5.2

import numpy as np
# initialize arrays
A = np . array ([2 , 1 , 5 , 4])
B = np . array ([3 , 4 , 7 , 8])
#dot product
output = np . dot(A , B )
print ( output )


5.3

import numpy as np

# initialize arrays
A = np.array([2, 1, 5, 4])
B = np.array([3, 4, 7, 8])

# dot product
output = np.dot(A, B)
print('Inner product is :', output)

if output == 0:
    print('given vectors are orthogonal')
else:
    print('given vectors are not orthogonal')


Lab 6

6.2.1

from sympy import *

x = Symbol('x')
g = input('Enter the function ')  # %x^3-2*x-5; % function
f = lambdify(x, g)

a = float(input('Enter a value :'))  # 2
b = float(input('Enter b value :'))  # 3
N = int(input('Enter number of iterations :'))  # 5

for i in range(1, N + 1):
    c = (a * f(b) - b * f(a)) / (f(b) - f(a))
    if (f(a) * f(c) < 0):
        b = c
    else:
        a = c
    print('iteration %d \t the root %0.3f \t function value %0.3f \n' %
          (i, c, f(c)))

6.2.2

from sympy import *

x = Symbol('x')
g = input('Enter the function ')  # Example: 'x**3 - 2*x - 5'
f = lambdify(x, g)

a = float(input('Enter a value :'))  # Example: 2
b = float(input('Enter b value :'))  # Example: 3
N = float(input('Enter tolerance :'))  # Example: 0.001

# Initialize x and c for the while loop condition
# x will hold the previous root approximation, c will hold the current
x_prev = a
c = b 
i = 0

while (abs(x_prev - c) >= N):
    x_prev = c
    c = ((a * f(b) - b * f(a)) / (f(b) - f(a)))
    
    if ((f(a) * f(c) < 0)):
        b = c
    else:
        a = c
    
    i = i + 1
    print('iteration %d \t the root %0.3f \t function value %0.3f \n' %
          (i, c, f(c)))

print('final value of the root is %0.5f ' % c)


6.3

from sympy import *

x = Symbol('x')
g = input('Enter the function ')  # Example: '3*x - cos(x) - 1'
f = lambdify(x, g)

dg = diff(g)
df = lambdify(x, dg)

x0 = float(input('Enter the initial approximation :'))  # Example: 1
n = int(input('Enter the number of iterations :'))  # Example: 5

for i in range(1, n + 1):
    x1 = (x0 - (f(x0) / df(x0)))
    print('iteration %d \t the root %0.3f \t function value %0.3f \n' %
          (i, x1, f(x1)))  # print all iteration values
    x0 = x1


Lab 8

def my_func(x):
    return 1 / (1 + x**2)

def simpson13(x0, xn, n):
    h = (xn - x0) / n  # calculating step size

    # Finding sum
    integration = (my_func(x0) + my_func(xn))

    # Correcting the loop and function calls for Simpson's 1/3 rule
    for i in range(1, n):
        k = x0 + i * h  # Calculate x_i
        if i % 2 == 0:
            integration = integration + 2 * my_func(k)
        else:
            integration = integration + 4 * my_func(k)

    # Finding final integration value
    integration = integration * h * (1 / 3)
    return integration

# Input section
lower_limit = float(input("Enter lower limit of integration : "))
upper_limit = float(input("Enter upper limit of integration : "))
sub_interval = int(input("Enter number of sub intervals : "))

# Ensure an even number of sub-intervals for Simpson's 1/3 rule
if sub_interval % 2 != 0:
    print("Number of sub-intervals must be even for Simpson's 1/3 rule.")
    # You might want to exit or adjust sub_interval here
    # For now, let's proceed with the given odd number and see the result,
    # but be aware it's not strictly correct for Simpson's 1/3.
    # A more robust solution would be to increment sub_interval by 1
    # or prompt the user again.

# Call simpson13() method and get result
result = simpson13(lower_limit, upper_limit, sub_interval)
print("Integration result by Simpson's 1/3 method is: %0.6f" % (result))


Lab 10

from sympy import *
import numpy as np

def RungeKutta(g, x0, h, y0, xn):
    x, y = symbols('x,y')
    f = lambdify([x, y], g)

    xt = x0 + h
    Y = [y0]

    while xt <= xn + 1e-9:  # Added a small epsilon for float comparison
        k1 = h * f(x0, y0)
        k2 = h * f(x0 + h / 2, y0 + k1 / 2)
        k3 = h * f(x0 + h / 2, y0 + k2 / 2)
        k4 = h * f(x0 + h, y0 + k3)

        y1 = y0 + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        Y.append(y1)

        x0 = xt
        y0 = y1
        xt = xt + h
    return np.round(Y, 2)

RungeKutta('1+(y/x)', 1, 0.2, 2, 2)
