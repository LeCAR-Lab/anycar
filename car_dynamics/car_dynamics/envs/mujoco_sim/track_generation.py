import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
import random
import math

def generate_random_points(num_points = 10, min_distance = 20):
    points = []
    for i in range(num_points):
        x = random.randrange(0,  100, 1)
        y = random.randrange(0, 100 , 1)
        distances = list(filter(lambda x: x < min_distance, [math.sqrt((p[0]-x)**2 + (p[1]-y)**2) for p in points]))
        if len(distances) == 0:
            points.append((x, y))
    return np.array(points)

def check_min_distances(og_points, min_distance):
    points = []
    for [x,y] in og_points:
        distances = list(filter(lambda x: x < min_distance, [math.sqrt((p[0]-x)**2 + (p[1]-y)**2) for p in points]))
        if len(distances) == 0:
            points.append((x, y))
    return np.array(points)

def get_track_points(hull, points):
    # get the original points from the random 
    # set that will be used as the track starting shape
    return np.array([points[hull.vertices[i]] for i in range(len(hull.vertices))])

def expand_track(points):
    new_points = []
    num_points = points.shape[0]
    for i in range(-1, num_points - 1): # should start from -1
        curr_x = points[i][0]
        curr_y = points[i][1]
        next_x = points[i+1][0]
        next_y = points[i+1][1]
        new_points.append([curr_x, curr_y])

        mid_x = (curr_x + next_x) / 2
        mid_y = (curr_y + next_y) / 2

        a = next_x - curr_x
        b = next_y - curr_y
    
        # Find an orthogonal vector
        ortho_vec = np.array([-b, a])
        if np.linalg.norm(ortho_vec) > 10:
            norm_ortho_vec = ortho_vec/np.linalg.norm(ortho_vec)
            random_distance = np.random.uniform(10, 15)
            sign = 1 if random.random() < 0.5 else -1

            transformed_point = sign * random_distance * norm_ortho_vec + np.array([mid_x, mid_y])

            new_points.append([transformed_point[0], transformed_point[1]])
    
    return np.array(new_points)

def fit_spline(points, timesteps):
    points = np.vstack((points[-1], points))
    tck, u = splprep([points[:,0], points[:,1]], s=0)
    u_new = np.linspace(0, 1, timesteps)
    x_new, y_new = splev(u_new, tck)
    return(np.hstack((np.array(x_new).reshape(-1, 1), np.array(y_new).reshape(-1,1))))
