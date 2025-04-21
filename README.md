# particle-tracking

## Description
A modified linear assignment problem based single-particle tracking algorithm from [Jaqaman et al. (2008)]([doi.org/10.1016/S0031-3203(01)00127-3](https://doi.org/10.1038/nmeth.1237)). In short, this algorithm considers both spatial distance and particle intensity to predict a particle's track, which allows for higher accuracy. Additionally, the algorithm has been optimized for efficiency. For more details, please read [methods](method.pdf). For an example tracking, please refer to the following [gif](example/tracked_simulation.gif).


## Procedure
Run "pip install -r requirements.txt" on your terminal if you don't have the following libraries: matplotlib, numpy, opencv_python, pandas, Pillow, and scipy.

The general procedure to use the single-particle tracking algorithm (look at [example](example/) for guidance):
  1. Load in your movie csv file. It should have the following columns named as (case senstive): frame, position_i, position_j, and intensity.
  2. Create ParticleTracking(movie_df) object.
     * **movie_df** is the movie csv that you loaded.
  3. Call predict_track(alpha, beta, max_sq_dist, gap_max_sq_dist, frame_window).
     * **alpha** is the weight for the squared Euclidean distance term. Generally alpha = 0.9 works well.
     * **beta** is the weight for squared particle intensity difference term. Generally beta = 0.1 works well.
     * **max_sq_dist** is the maximum squared Euclidean distance to consider for particle linking.
     * **gap_max_sq_dist** is the maximum squared Euclidean distance to consider for gap closing.
     * **frame_window** is the maximum number of frames a particle can disappear before it won't be considered for gap closing.
  4. Call estimate_ratio().
     * Check if the ratio is >1 to ensure that particle intensity is a good predictor of a particle's track. Otherwise run the algorithm again but set alpha = 1 and beta = 0. 
  5. Call animate_particle_tracks(images, save_path).
     * **images** is a N by (H, W) numpy array that stores the images used to create the movie csv (N = number of frames, H = image height, W = image width).
     * **save_path** is the the path where you want to save the ".gif" file. If None, then file will not save and you can interactively visualize the particle tracks.
  6. Call save_tracks(path_name).
     * **path_name** is the the path where you want to save the predicted tracks as a csv file.

