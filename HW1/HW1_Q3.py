import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


sys.stdout = open("Homework1_Q3_output.txt", "w")

# Code Referred from 2_2 (3Dto3DTransforms), 2_3 (3Dto2DTransforms) slides

### VEHICLE {V} TO WORLD {W}
#Rotation of vehicle relative to world coordinates
az = np.pi/6 # 30 degrees in radians
sz = np.sin(az)
cz = np.cos(az)
Rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
# Translation: origin of V in W.
tVorg_W = np.array([[6,-8,1]]).T
# H_V_W means transform V to W.
H_V_W = np.block([[Rz, tVorg_W], [0,0,0,1]])

### MOUNT {M} TO VEHICLE {V}
#Rotation of mount relative to vehicle coordinates
ax = - 2*np.pi/3 # -120 degrees in radians
sx= np.sin(ax)
cx = np.cos(ax)
Rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
# Translation: origin of M in V.
tMorg_V = np.array([[0,0,3]]).T
# H_M_V means transform M to V.
H_M_V = np.block([[Rx, tMorg_V], [0,0,0,1]])

### CAMERA {C} TO MOUNT {M}
I = np.identity(3) # no rotation for this case, just identity matrix
# Translation: origin of C in M.
tCorg_M = np.array([[0,-1.5,0]]).T
# H_C_M means transform C to M.
H_C_M = np.block([[I, tCorg_M], [0,0,0,1]])

### WORLD {W} to CAMERA {C}
H_C_W = H_V_W @ H_M_V @ H_C_M
H_W_C = np.linalg.inv(H_C_W)

### CAMERA
# Intrisic Calibration Matrix
f = 600.0 # focal length in pixels
sx = 1
sy = 1
cx = 320
cy = 240
K = np.array(((f/sx, 0, cx), (0, f/sy, cy), (0, 0, 1)))
# Extrinsic Matrix
Mext = H_W_C[0:3, :]

### PYRAMID
# In World Coordinates
P_w = np.array([[-1, 1, 1, -1, 0],
                [-1, -1, 1, 1, 0],
                [0, 0, 0, 0, 3],
                np.ones(5)])
# In Camera Coordinates
p_C = K @ Mext @ P_w
p_C = p_C / p_C[2,:] # Keeping in Homogeneous Coordinates
print(np.rint(p_C[0:2, :]))

### Creating Wireframe image of Pyramid
Image = 255*np.ones((480, 640), dtype=np.uint8)

# Adding Wireframe to adjacent points
for i in range(4):
    cv2.line(Image, (p_C[0,i].astype(int), p_C[1,i].astype(int)), (p_C[0,i+1].astype(int), p_C[1,i+1].astype(int)), 0, thickness=2)

#Adding Wireframe to other 3 base points to vertex of pyramid
cv2.line(Image, (p_C[0,0].astype(int), p_C[1,0].astype(int)), (p_C[0,4].astype(int), p_C[1,4].astype(int)), 0, thickness=2)
cv2.line(Image, (p_C[0,1].astype(int), p_C[1,1].astype(int)), (p_C[0,4].astype(int), p_C[1,4].astype(int)), 0, thickness=2)
cv2.line(Image, (p_C[0,2].astype(int), p_C[1,2].astype(int)), (p_C[0,4].astype(int), p_C[1,4].astype(int)), 0, thickness=2)
# Connecting First and Fourth base corners
cv2.line(Image, (p_C[0,0].astype(int), p_C[1,0].astype(int)), (p_C[0,3].astype(int), p_C[1,3].astype(int)), 0, thickness=2)

cv2.imshow("Pyramid Wireframe", Image)
cv2.imwrite("Homework1_PyramidWireframe.jpg", Image)
cv2.waitKey(0)


### Plotting 3D scene, Code used from 2-4, TransformsAdditional slides

# Draw three 3D line segments, representing xyz unit axes, onto the axis figure ax.
# H is the 4x4 transformation matrix representing the pose of the coordinate frame.
def draw_coordinate_axes(ax, H, label):
    p = H[0:3, 3]      # Origin of the coordinate frame
    ux = H @ np.array([1,0,0,1])# Tip of the x axis
    uy = H @ np.array([0,1,0,1])# Tip of the y axis
    uz = H @ np.array([0,0,1,1])# Tip of the z axis
    ax.plot(xs=[p[0], ux[0]], ys=[p[1], ux[1]], zs=[p[2], ux[2]], c='r')# x axis
    ax.plot(xs=[p[0], uy[0]], ys=[p[1], uy[1]], zs=[p[2], uy[2]], c='g')# y axis
    ax.plot(xs=[p[0], uz[0]], ys=[p[1], uz[1]], zs=[p[2], uz[2]], c='b')# z axis
    ax.text(p[0], p[1], p[2], label)   # Also draw the label of the coordinate frame

# Utility function for 3D plots.
def setAxesEqual(ax):
    # '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    # cubes as cubes, etc..  This is one possible solution to Matplotlib's
    # ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    # Input      ax: a matplotlib axis, e.g., as output from plt.gca().
    # '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

#creating Pyramid coordinates such that we can form wireframe using 3D plot
Pyramid = np.c_[P_w, P_w[:,2], P_w[:,4], P_w[:,1], P_w[:,4], P_w[:,0], P_w[:,3]]

#Creating Figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(Pyramid[0,:], Pyramid[1,:], zs=Pyramid[2,:])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Scene of Vehicle, Mount, and Camera in World with Pyramid')
ax.view_init(20, 30)
draw_coordinate_axes(ax, np.eye(4), 'W') #world
draw_coordinate_axes(ax, H_C_W, 'C') # camera pose in world
draw_coordinate_axes(ax, H_V_W, 'V') # vehicle pose in world
draw_coordinate_axes(ax, H_V_W @ H_M_V, 'M') # mount pose in world
setAxesEqual(ax)
fig.savefig("Homework1_3DScene.jpg")
plt.show()  # This shows the plot, and pauses until you close the figure


sys.stdout.close()
