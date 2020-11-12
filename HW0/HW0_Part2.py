import numpy as np
import sys


def gaussian9by9matrix(xymean, sigma):
    g = np.zeros((9, 9))
    x = np.linspace(xymean[0] - 4, xymean[0] + 4, 9)
    y = np.linspace(xymean[1] - 4, xymean[1] + 4, 9)
    for i in range(9):
        for j in range(9):
            # using 2D Gaussian distribution as given in https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
            g[i, j] = np.exp(-((x[i]-xymean[0])** 2 + (y[j]-xymean[1])** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma** 2)

    # Verify sum of matrix elements add up to 1
    if round(np.sum(g)) == 1:
        print("Great! For mean = (%d,%d) and sigma = %f, the sum of the matrix values is %f, close to 1." %(xymean[0], xymean[1], sigma, np.sum(g)))
    elif round(np.sum(g)) > 1:
        print("For mean = (%d,%d) and sigma = %f, the sum of the matrix values is %f. Try again with a larger sigma." %(xymean[0], xymean[1], sigma, np.sum(g)) )
    else:
        print("For mean = (%d,%d) and sigma = %f, the sum of the matrix values is %f. Try again with a smaller sigma." %(xymean[0], xymean[1], sigma, np.sum(g)))
    return g


sys.stdout = open("HW0_Part2_output.txt", "w")

print("==== Sigma = 1 ====")
G = gaussian9by9matrix((0, 0), 1)
print("Output matrix for mean=(0,0) and sigma = 1:")
print(G)

print("Mean values different from (0,0), verifying that matrix elements still add up to 1:")
G = gaussian9by9matrix((2, 2), 1)
G = gaussian9by9matrix((-1, 2), 1)

print("==== Sigma != 1 ====")
gaussian9by9matrix((0, 0), 5)
gaussian9by9matrix((0, 0), 0.1)

print("Mean values different from (0,0), verifying that matrix elements do not add up to 1 when sigma != 1:")
G = gaussian9by9matrix((2, 2), 5)
G = gaussian9by9matrix((-1, 2), 0.1)

sys.stdout.close()
