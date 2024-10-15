import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
import random
import math
import pandas as pd

def generate_random_points(num_points = 10, scale = 1):
    points = []
    min_distance = int(4 * scale)
    for i in range(num_points):
        x = random.randrange(0,  int(20 * scale), 1)
        y = random.randrange(0, int(20 * scale), 1)
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

def expand_track(points, scale):
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
        if np.linalg.norm(ortho_vec) > 4:
            norm_ortho_vec = ortho_vec/np.linalg.norm(ortho_vec)
            random_distance = np.random.uniform(2*scale, 5*scale)
            sign = 1 if random.random() < 0.5 else -1

            transformed_point = sign * random_distance * norm_ortho_vec + np.array([mid_x, mid_y])

            new_points.append([transformed_point[0], transformed_point[1]])
    
    return np.array(new_points)

def fit_spline(points, timesteps):
    points = np.vstack((points[-1], points))
    tck, u = splprep([points[:,0], points[:,1]], s=0, per=True)
    u_new = np.linspace(0, 1, timesteps)
    x_new, y_new = splev(u_new, tck)
    return(np.hstack((np.array(x_new).reshape(-1, 1), np.array(y_new).reshape(-1,1))))

def change_track(scale:int = 1, direction = 1):
    num_points = 5 * max(scale, 1)
    num_total_points = 100 * max(scale, 1)
    random_points = generate_random_points(num_points, scale)
    # get convex hull

    hull = None
    while hull is None:
        try:
            hull = ConvexHull(random_points)
        except:
            random_points = generate_random_points(num_points)
            pass

    track_points = get_track_points(hull, random_points)
    expanded_track_points = expand_track(track_points, scale)
    fitted_spline = fit_spline(expanded_track_points, num_total_points)
    if direction == -1:
        fitted_spline = np.flip(fitted_spline)
    # plt.plot(fitted_spline[:,0], fitted_spline[:,1])
    # plt.show()
    return fitted_spline


if __name__ == "__main__":
        track = change_track(5, 1)
        print("track shape", track.shape)
        plt.plot(track[:,0], track[:,1])
        plt.show()
        # df = pd.DataFrame(track, columns = ["x", "y"])
        # df.to_csv("track2.csv", index = False, header = False)

        