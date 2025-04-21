import os
import cv2
import numpy as np
import pandas as pd
from particle_tracking import ParticleTracking

# load in your movie dataframe
movie_df = pd.read_csv("simulation.csv")

# create ParticleTracking Object
movieInfo = ParticleTracking(movie_df)

# predict the tracks
movieInfo.predict_track(alpha=0.9, beta=0.1, max_sq_dist=900, gap_max_sq_dist=900, frame_window=2)

# estimates ratio
print(movieInfo.estimate_ratio())

# visualize tracks
files = os.listdir('frames')
if ".DS_Store" in files:
    files.remove(".DS_Store")
files.sort()
images = [cv2.imread(os.path.join('frames', file), cv2.IMREAD_ANYDEPTH) for file in files]

# set save_path to None to interactively visualize the tracks
movieInfo.animate_particle_tracks(images=np.array(images), save_path="tracked_simulation.gif")

# save track results
movieInfo.save_tracks(path_name="tracked_simulation.csv")