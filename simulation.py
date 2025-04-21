import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter


def simulation(
    num_frames,
    region_size,
    num_particles,
    diffusion_coeff,
    intensity_mean,
    intensity_sd,
    noise_sd,
):
    """
    Returns the dataframe of the simluation
    """
    # Particle ID and Frame
    particle_id = np.arange(num_particles)
    frame = np.zeros(num_particles, dtype=int)

    # Initialize particle positions randomly within the image
    i_positions = np.random.uniform(0, region_size[0], num_particles)
    j_positions = np.random.uniform(0, region_size[1], num_particles)

    # Base amplitude for each particle
    base_amplitudes = np.random.normal(intensity_mean, intensity_sd, num_particles)

    # Store output data
    records = pd.DataFrame(
        columns=["frame", "position_i", "position_j", "intensity", "track_id"]
    )

    # Brownian motion simulation
    for _ in range(num_frames):
        # vary the amplitudes
        amplitude = base_amplitudes + np.random.normal(0, noise_sd, num_particles)

        # record data
        new_df = pd.DataFrame(
            {
                "frame": frame,
                "position_i": i_positions,
                "position_j": j_positions,
                "intensity": amplitude,
                "track_id": particle_id,
            }
        )
        records = pd.concat([records, new_df], ignore_index=True)

        # Brownian motion step (Gaussian-distributed step size)
        i_positions += np.random.normal(0, np.sqrt(2 * diffusion_coeff), num_particles)
        j_positions += np.random.normal(0, np.sqrt(2 * diffusion_coeff), num_particles)

        # Keep within image bounds
        i_positions = np.clip(i_positions, 0, region_size[0] - 1)
        j_positions = np.clip(j_positions, 0, region_size[1] - 1)

        # Update frame number
        frame += 1

    return records


def create_gif(df, shape, sigma, duration, output_path):
    """
    Creates a gif of the simulation from the simulation dataframe.

    Inputs:
    df: simulation dataframe
    shape: image shape
    sigma: radius of the particles
    duration: duration of the gif in 10ths of a second
    output_path: the path to save the gif
    """
    def create_gaussian_spot(image_shape, i, j, intensity, sigma):
        blank_image = np.zeros(image_shape)
        i, j = int(round(i)), int(round(j))
        blank_image[i, j] += intensity
        return gaussian_filter(blank_image, sigma=sigma)

    frames = []

    for f in range(df["frame"].max() + 1):
        image = np.zeros(shape, dtype=np.float32)
        current = df[df["frame"] == f]

        for _, row in current.iterrows():
            i, j = row["position_i"], row["position_j"]
            intensity = row["intensity"]
            image += create_gaussian_spot(shape, i, j, intensity, sigma)

        # Normalize and convert to uint8
        norm_img = 255 * (image / image.max()) if image.max() > 0 else image
        frames.append(Image.fromarray(norm_img.astype(np.uint8)))

    frames[0].save(
        output_path, save_all=True, append_images=frames[1:], duration=duration, loop=0
    )


simulation_df = simulation(
    num_frames=100,
    region_size=(256, 256),
    num_particles=150,
    diffusion_coeff=1,
    intensity_mean=100,
    intensity_sd=5,
    noise_sd=1,
)

# Saving results
simulation_df.to_csv("simulation.csv", index=False)
create_gif(
    simulation_df, shape=(257, 257), sigma=2, duration=250, output_path="simulation.gif"
)
