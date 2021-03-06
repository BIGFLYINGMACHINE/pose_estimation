import cv2
import math
import numpy as np
import OpenPoseImage as opi
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
im = cv2.imread("mask.jpg")
size = im.shape

# 2D image points. If you change the image, you need to change vector
image_points = np.array(opi.get_location(im), dtype="double")

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),  # 肋骨交界处，胸口
    (-70, 0, 250.0),  # 锁骨交界处再往上40mm
    (-85, 142.0, 170.0),  # 左肩
    (-85.0, -142.0, 170),  # 右肩
    (-42, -120.0, -240.0),  # 右髂前上棘
    (-42, 120, -240.0)  # 左髂前上棘

])

# Camera internals

focal_length = (size[0]+size[1])/2
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)

print("Camera Matrix :\n {0}".format(camera_matrix))

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                                                              flags=cv2.SOLVEPNP_ITERATIVE)

print ("Rotation Vector:\n {0}".format(rotation_vector))
print ("Translation Vector:\n {0}".format(translation_vector))

# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose


(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector,
                                                 camera_matrix, dist_coeffs)

for p in image_points:
    cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

p1 = (int(image_points[0][0]), int(image_points[0][1]))
p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

# cv2.line(im, p1, p2, (255, 0, 0), 2)

# Display image
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
cv2.imshow("Output", im)
cv2.waitKey(0)

scale_factor = 1
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title("Classroom")
plt.xlim([-3600*scale_factor, 3600*scale_factor])
plt.ylim([0, 9000*scale_factor])
width = 500
height = 1000
human_x = int(translation_vector[0][0])
human_y = int(abs(translation_vector[2][0]))
print(human_x, human_y)
plt.plot([0, 0], [0, 9000], color='blue')
plt.plot([-3600, 3600], [3000, 3000], color='blue')
plt.plot([-3600, 3600], [6000, 6000], color='blue')
for x in [-1800, 1800]:
    for y in [1500, 4500, 7500]:
        x_lf = x - width/2
        y_lf = y - height/2
        if math.sqrt(math.pow(human_x-x_lf, 2) + math.pow(human_y-y_lf,2))< 2350:
            rect = Rectangle((x_lf, y_lf), width, height, color='yellow')
        else:
            rect = Rectangle((x_lf, y_lf), width, height, color='black')
        ax.add_patch(rect)
rect = Circle((human_x, human_y), 50, color = 'red')
ax.add_patch(rect)
plt.show()