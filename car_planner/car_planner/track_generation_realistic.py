import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
import random
import math
import pandas as pd


def change_track_feasible(scale:int = 1, direction = 1,  default_size = 4):

    def generate_random_points(num_points = 10, scale = 1):
        points = []
        for _ in range(num_points):
            x = random.randrange(0,  int(default_size * scale), 1)
            y = random.randrange(0, int(default_size * scale), 1)
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
            if np.linalg.norm(ortho_vec) > min_distance/2:
                norm_ortho_vec = ortho_vec/np.linalg.norm(ortho_vec)
                random_distance = np.random.uniform(default_size/10*scale, default_size/4*scale)
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
    
    def is_invalid_track(spline_points):
        for i in range(len(spline_points)):
            for j in range(i + 2, len(spline_points) - 1):
                if intersect(spline_points[i], spline_points[i+1], spline_points[j], spline_points[j+1]):
                            return True
        wheelbase = 0.3
        max_steer = 0.36
        max_curvature = np.tan(max_steer) / wheelbase
        curvature = compute_curvature(spline_points[:,0], spline_points[:,1])
        # print("actual max", max_curvature, "track max curve", curvature.max())
        if curvature.max() > max_curvature:
            return True
        
        return False
    
    def compute_curvature(x, y):
        """Compute the curvature at each point of the given x and y coordinates."""
        dx_dt = np.gradient(x)
        dy_dt = np.gradient(y)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        
        curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt**2 + dy_dt**2)**1.5
        return curvature

                
    def intersect(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        # Check if the line segments straddle each other
        if ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4):
            # If they do, calculate the intersection point
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            if denom == 0:  # Lines are parallel
                return False
            
            ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
            if ua < 0 or ua > 1:  # Intersection point isn't within the first line segment
                return False
            
            ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
            if ub < 0 or ub > 1:  # Intersection point isn't within the second line segment
                return False
            
            return True
        
        return False
         
    num_points = 10 * max(scale, 1)
    num_total_points = 100 * max(scale, 1)
    min_distance = int(default_size/4 * scale)
    not_found_valid_track = True
    iters = 0
    while not_found_valid_track:
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
        if not is_invalid_track(fitted_spline):
            print(f"Found valid track in {iters} iterations")
            not_found_valid_track = False
        iters += 1
    # plt.plot(random_points[:,0], random_points[:,1], ".", label="random points")
    # plt.plot(expanded_track_points[:,0], expanded_track_points[:,1], "x", label = "expanded track points")
    if direction == -1:
        fitted_spline = np.flip(fitted_spline)
    # plt.plot(fitted_spline[:,0], fitted_spline[:,1])
    # plt.show()
    
    upper_point = fitted_spline[:, 1].argmax()
    lower_point = fitted_spline[:, 1].argmin()
    
    if direction == -1:
        # clockwise
        fitted_spline[:, 1] -= fitted_spline[upper_point, 1]
        fitted_spline[:, 0] -= fitted_spline[upper_point, 0]
    else:
        # counter clockwise
        fitted_spline[:, 1] -= fitted_spline[lower_point, 1]
        fitted_spline[:, 0] -= fitted_spline[lower_point, 0]
    
    return fitted_spline


if __name__ == "__main__":
        # plot 25 tracks in a figure, by 5 times 5 subfigures, each figure is a track
        plt.figure(figsize=(20, 20))
        for i in range(25):
            print(i)
            plt.subplot(5, 5, i+1)
            # scale = int(np.random.uniform(1, 4))
            scale = 1
            direction = np.random.choice([-1, 1])
            track = change_track(scale, direction)
            if direction == -1:
                color = "red"
                label = "clockwise"
            else:
                color = "blue"
                label = "counter clockwise"
            plt.plot(track[:,0], track[:,1], color = color, label=label)
            plt.legend()
        # plt.show()
        plt.savefig("track.png")
        # scale = 1
        # direction = 1
        # track = change_track(scale, direction)
        # print("track shape", track.shape)
        # plt.plot(track[:,0], track[:,1])
        # plt.legend()
        # plt.savefig("track.png")
        # plt.show()
        # df = pd.DataFrame(track, columns = ["x", "y"])
        # df.to_csv("track2.csv", index = False, header = False)

        