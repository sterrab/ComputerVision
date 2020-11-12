import numpy as np
import sys

sys.stdout = open("HW0_Part1_output.txt", "w")

### Question 1:
A = np.array([[4, -2], [1, 1]])
print("===QUESTION 1 ANSWERS===")

det_A = np.linalg.det(A)
print("The determinant of A is %1.1f." % det_A)

trace_A = np.trace(A)
print("The trace of A is %1.1f." % trace_A)

inv_A = np.linalg.inv(A)
print("The inverse of A is: ")
print(inv_A)

evals, evectors = np.linalg.eig(A)
evector1_nonnormalized = evectors[:,0]/min(evectors[:,0])
print("One of the eigenvectors of A  is: ")
print(evector1_nonnormalized[:,None])
print("Its corresponding eigenvalue is %1.1f." % evals[0])

### Question 2:

B = np.array([[3, 4], [5, -1]])
print("===QUESTION 2 ANSWERS===")

print("(AB)^T : ")
print(np.transpose(A@B))

print("B^TA^T : ")
print(np.transpose(B)@np.transpose(A))

### Question 3:

x = np.array([1,2,3])
y = np.array([-1,2,-3])
print("===QUESTION 3 ANSWERS===")

xdoty = np.inner(x,y)
print("The inner product of x and y is %1.1f." %xdoty)

print("The vector product of x and y is: ")
print(np.cross(x,y))

sys.stdout.close()