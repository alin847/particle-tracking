from data_structure import LinkedList
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm


class ParticleTracking:
    def __init__(self, movie_df):
        # checking requirements
        required_cols = ['frame', 'position_i', 'position_j', 'intensity']
        missing_cols = [col for col in required_cols if col not in movie_df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # track dataframe
        self.track_df = movie_df.copy()

        # internal housekeeping for optimization
        self.frame_index = self.create_frame_index(movie_df)
        self.tracks = []
        self.incrementer_gen = self.counter(len(self.frame_index[0]))


    # MAIN
    def predict_track(self, alpha, beta, max_sq_dist, gap_max_sq_dist, frame_window):
        """
        Predicts the tracks given the movie dataframe. Updates the track_df.

        Inputs:
        alpha: weight for squared distance
        beta: weight for squared intensity difference
        max_sq_dist: maximum square distance to consider for frame linking
        gap_max_sq_dist: maximum square distance to consider for gap closing
        frame_window: number of frames to look fowards
        """
        if alpha + beta != 1:
            raise ValueError("alpha + beta must equal 1")
        
        self.start_track()
        current_tracks = self.tracks

        for i in range(max(self.frame_index)):
            current_values = self.get_frame_particles(i)
            next_values = self.get_frame_particles(i+1)
            cost_matrix = self.create_cost_matrix(current_values, next_values, alpha, beta, max_sq_dist)

            # transpose so that we have the assignments in correct format
            _, assignments = linear_sum_assignment(cost_matrix.transpose())
            current_tracks = self.update_track(assignments, current_tracks, self.frame_index[i+1])

        start_tracks = self.get_start_tracks()
        end_tracks = self.get_end_tracks()

        gap_matrix = self.create_gap_matrix(end_tracks, start_tracks, alpha, beta, gap_max_sq_dist, frame_window)
        _, assignments = linear_sum_assignment(gap_matrix.transpose())

        self.gap_closing(assignments)
    
    
    def estimate_ratio(self):
        """
        Estimates the ratio between the particle intensity and noise standard deviation.
        """
        if not self.tracks:
            raise RuntimeError("You must call 'predict_track()' first.")
        
        noise_var = np.empty(len(self.tracks))
        observed_var = np.empty(len(self.frame_index))

        for i, track in enumerate(self.tracks):
            noise_var[i] = np.var(self.track_df.loc[track.to_list(), 'intensity'].values)
        
        for i, indices in self.frame_index.items():
            observed_var[i] = np.var(self.track_df.loc[indices, 'intensity'].values)
        
        mean_noise_var = np.mean(noise_var)
        mean_observed_var = np.mean(observed_var)

        return np.sqrt( (mean_observed_var - mean_noise_var) / mean_noise_var)


    def animate_particle_tracks(self, images, save_path=None):
        """
        Inputs:
        images: a N by H by W array where N = number of frames, H = height,
        and W = width
        save_path: name of the file to so save. If None, then does not save.
        """
        num_frames = images.shape[0]
        track_ids = np.arange(len(self.tracks))
        cmap = cm.get_cmap('tab20', len(track_ids))
        color_map = {tid: cmap(i) for i, tid in enumerate(track_ids)}

        fig, ax = plt.subplots(figsize=(8, 8))
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        img_display = ax.imshow(images[0], cmap='gray', origin='upper')
        scatter_plots = {}  # track_id: plt.Line2D

        # Initialize empty scatter plot per track
        for track_id in track_ids:
            scatter, = ax.plot([], [], 'o-', color=color_map[track_id], markersize=1, linewidth=0.5)
            scatter_plots[track_id] = scatter

        def update(frame_idx):
            ax.set_title(f'Frame {frame_idx}')
            img_display.set_array(images[frame_idx])

            frame_data = self.track_df[self.track_df['frame'] <= frame_idx]

            for track_id, scatter in scatter_plots.items():
                track = frame_data[frame_data['track_id'] == track_id]
                scatter.set_data(track['position_j'], track['position_i'])

            return [img_display] + list(scatter_plots.values())

        ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=500, blit=True)

        if save_path:
            ani.save(save_path, writer='pillow', fps=5)
        else:
            plt.show()


    def save_tracks(self, path_name):
        """Saves the track_df to path_name"""
        self.track_df.to_csv(path_name, index=False)

        
    # HELPERS
    def counter(self, n):
        while True:
            yield n
            n += 1


    def create_frame_index(self, movie_df):
        """
        Creates the frame index dictionary for faster look up time.
        Key corresponds to the frame and value corresponds to the list
        of indices.

        dictionary = {
        0: [0,1,...],
        1: [15,16,...],
        .
        .
        .
        frame_N: [200,201,...]
        }
        """
        def add_to_frame_index(frame, index):
            if frame in frame_index:
                frame_index[frame].append(index)
            else:
                frame_index[frame] = [index]
        

        frame_index = {}
        for i, frame in enumerate(movie_df["frame"]):
            add_to_frame_index(int(frame), i)
        return frame_index


    def start_track(self):
        """
        Starts the track by initializing each particle in first frame 
        as an unique track.
        """
        indices = self.frame_index[0]
        for index in indices:
            track = LinkedList()
            track.append(index)
            self.tracks.append(track)


    def update_track(self, next_frame_assignments, current_frame_tracks, next_frame_indices):
        """
        Inputs:
        next_frame_assignments: the assignment of each next frame from the lap
        assignment in order. For example, if the array was [3,1,4,5,2,0], then
        particle 0 in next frame is assigned to particle 3 in current frame.
        Particle 1 in next frame is assigned to particle 1 in current frame and
        so on.
        current_frame_tracks: a list of tracks (linked lists) of the current frame,
        ordered from particle0, particle1, ... from current frame
        next_frame_indices: indices of the next frame from the frame_index 
        dictionary.

        Outputs:
        next_frame_tracks: a list of tracks (linked lists) for the next iteration
        for update track.
        """
        next_frame_tracks = []

        for i, row in enumerate(next_frame_indices):
            if next_frame_assignments[i] < len(current_frame_tracks):
                # update track
                current_frame_tracks[next_frame_assignments[i]].append(row)
                next_frame_tracks.append(current_frame_tracks[next_frame_assignments[i]])
            else:
                # start new track
                new_track = LinkedList()
                new_track.append(row)
                self.tracks.append(new_track)
                next_frame_tracks.append(new_track)

        return next_frame_tracks
    

    def gap_closing(self, assignments):
        """
        Given the lap_solution to the gap_closing cost matrix, updates all the track_ids
        to close gaps in the track_df.
        
        Inputs:
        assignment: the assignment of each end frame from the lap assignment 
        in order. For example, if the array was [3,2,4,5,1,0,6], then the start
        of track 0 is assigned to the end of track 3. Start of track 1 is
        assigned to end of track 2 and so on.
        """
        N_tracks = int(len(assignments)/2)
        relevant_assignments = assignments[:N_tracks]

        for start_track_id, end_track_id in reversed(list(enumerate(relevant_assignments))):
            if end_track_id < N_tracks:
                self.tracks[end_track_id].extend(self.tracks[start_track_id])
                del self.tracks[start_track_id]

        # setting my track_ids
        track_ids = np.full(len(self.track_df), -1, dtype=int)

        for track_id, track in enumerate(self.tracks):
            indices = track.to_list()
            track_ids[indices] = track_id
        
        self.track_df['track_id'] = track_ids
    

    def create_cost_matrix(self, current_frame, next_frame, alpha, beta, max_sq_dist):
        """
        Creates the cost matrix (in COO format) for the linear assignment problem.

        Inputs:
        current_frame: N1 by 3 array of (i, j, intensity) in frame t  
        next_frame: N2 by 3 array of (i, j, intensity) in frame t+1
        alpha: weight for squared distance
        beta: weight for squared intensity difference
        max_sq_dist: maximum squared distance to consider

        Outputs:
        cost_matrix: cost matrix for the LAP in COO format
        max_cost: new maximum cost of linking
        """
        N_current_frame = current_frame.shape[0]
        N_next_frame = next_frame.shape[0]

        # square distance constraint
        sq_dist_matrix = cdist(current_frame[:,:2], next_frame[:,:2], metric='sqeuclidean')
        frame_linking_rows, frame_linking_cols = np.where(sq_dist_matrix <= max_sq_dist)
        sq_dist_vals = sq_dist_matrix[frame_linking_rows, frame_linking_cols]

        # calculating linking block
        if len(sq_dist_vals) == 0:  # no frame linking block
            frame_linking_vals = np.array([])
        else:
            sq_intensity_vals = (next_frame[frame_linking_cols, 2] - current_frame[frame_linking_rows, 2]) ** 2
            sq_dist_zscores = self.z_scores(sq_dist_vals)
            sq_intensity_zscores = self.z_scores(sq_intensity_vals)
            frame_linking_vals = alpha * sq_dist_zscores + beta * sq_intensity_zscores

        # calculating other blocks
        max_cost = 1e2
        start_track_rows = N_current_frame + np.arange(N_next_frame)
        start_track_cols = np.arange(N_next_frame)
        start_track_vals = np.full(N_next_frame, max_cost)

        end_track_rows = np.arange(N_current_frame)
        end_track_cols = N_next_frame + np.arange(N_current_frame)
        end_track_vals = np.full(N_current_frame, max_cost)

        auxiliary_rows = N_current_frame + frame_linking_cols
        auxiliary_cols = N_next_frame + frame_linking_rows
        auxiliary_vals = np.full(frame_linking_vals.shape[0], -1e2)

        # creating cost matrix
        cost_rows = np.concatenate([frame_linking_rows, start_track_rows, end_track_rows, auxiliary_rows])
        cost_cols = np.concatenate([frame_linking_cols, start_track_cols, end_track_cols, auxiliary_cols])
        cost_vals = np.concatenate([frame_linking_vals, start_track_vals, end_track_vals, auxiliary_vals])
        cost_vals += 101  # make all costs positive

        cost_matrix = np.full((N_current_frame + N_next_frame, N_current_frame + N_next_frame), 1e9)
        cost_matrix[cost_rows, cost_cols] = cost_vals
        return cost_matrix


    def create_gap_matrix(self, end_tracks, start_tracks, alpha, beta, max_sq_dist, frame_window):
        """
        Creates the gap closing cost matrix in COO format for the LAP. 

        Inputs:
        end_tracks: N by 4 array of (frame, i, j, intensity) of end of track
        start_tracks: N by 4 array of (frame, i, j, intensity) of start of track
        alpha: weight for squared distance
        beta: weight for squared intensity difference
        max_sq_dist: maximum squared distance to consider
        frame_window: number of frames to look fowards

        Output:
        cost_matrix: the gap closing cost matrix
        """
        N_tracks = start_tracks.shape[0]

        # frame_window contraint
        frame_matrix = start_tracks[:, 0][np.newaxis, :] - end_tracks[:, 0][:, np.newaxis]
        frame_linking_rows, frame_linking_cols = np.where((frame_matrix > 0) & (frame_matrix <= frame_window))

        # max_sq_dist constraint
        sq_dist_vals = np.sum((end_tracks[frame_linking_rows, 1:3] - start_tracks[frame_linking_cols, 1:3])**2, axis=1)
        valid_index = np.where(sq_dist_vals <= max_sq_dist)[0]
        frame_linking_rows = frame_linking_rows[valid_index]
        frame_linking_cols = frame_linking_cols[valid_index]
        sq_dist_vals = sq_dist_vals[valid_index]

        # calculating gap closing block
        if len(sq_dist_vals) == 0:  # no gap closing block
            frame_linking_vals = np.array([])
        else:
            sq_intensity_vals = (end_tracks[frame_linking_rows, 3] - start_tracks[frame_linking_cols, 3]) ** 2
            sq_dist_zscores = self.z_scores(sq_dist_vals)
            sq_intensity_zscores = self.z_scores(sq_intensity_vals)
            frame_linking_vals = alpha * sq_dist_zscores + beta * sq_intensity_zscores

        # calculating other blocks
        max_cost = 1e2
        start_track_rows = np.arange(N_tracks) + N_tracks
        start_track_cols = np.arange(N_tracks)
        start_track_vals = np.full(N_tracks, max_cost)

        end_track_rows = np.arange(N_tracks)
        end_track_cols = np.arange(N_tracks) + N_tracks
        end_track_vals = np.full(N_tracks, max_cost)

        auxiliary_rows = N_tracks + frame_linking_cols
        auxiliary_cols = N_tracks + frame_linking_rows
        auxiliary_vals = np.full(frame_linking_vals.shape[0], -1e2)

        # creating the cost matrix
        cost_rows = np.concatenate([frame_linking_rows, start_track_rows, end_track_rows, auxiliary_rows])
        cost_cols = np.concatenate([frame_linking_cols, start_track_cols, end_track_cols, auxiliary_cols])
        cost_vals = np.concatenate([frame_linking_vals, start_track_vals, end_track_vals, auxiliary_vals])   
        cost_vals += 101  # make costs positive

        cost_matrix = np.full((2 * N_tracks, 2 * N_tracks), 1e9)
        cost_matrix[cost_rows, cost_cols] = cost_vals

        return cost_matrix


    def get_frame_particles(self, frame_number):
        """
        returns the (i, j, intensity) of all the particles in a frame as a
        N by 3 array.
        """
        indices = self.frame_index[frame_number]
        return self.track_df.loc[indices, ['position_i', 'position_j', 'intensity']].values


    def get_start_tracks(self):
        """
        returns the (frame, i, j, intensity) of the start of the track 
        in order of track_id as a N by 4 array.
        """
        start_indices = [track.get_head() for track in self.tracks]
        return self.track_df.loc[start_indices, ['frame', 'position_i', 'position_j', 'intensity']].values


    def get_end_tracks(self):
        """
        returns the (frame, i, j, intensity) of the end of the track 
        in order of track_id as a N by 4 array.
        """
        end_indices = [track.get_tail() for track in self.tracks]
        return self.track_df.loc[end_indices, ['frame', 'position_i', 'position_j', 'intensity']].values


    def z_scores(self, array):
        """Takes a 1D array an returns a new array after zscore normalization"""
        std = np.std(array)
        if std == 0:
            return np.zeros_like(array)
        else:
            return (array - np.mean(array)) / std

