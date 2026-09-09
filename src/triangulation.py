"""
This package provides tools for the localisation of a sound source within a
known volume, based on recordings from pairs of synchronised recorders.

Each pair of synchronised recorders constrains source position along a
revolution hyperboloid defined by the time difference of arrival of the sound.
This package provides two tools for TDOA measurement : cross_correlation and
gcc_phat.

The intersection of three revolution hyperboloids results in a few possible
solutions. These solutions should be symmetric along the hyperboloids, so
knowing the physical bounds of the volume of interest should reduce the
solutions to a single estimated position.
Although an analytical solution should be computable, this package relies on
the Gauss-Newton iterative algorithm to find the sound source position.

Main functions
--------------
- localise_3pairs
    Find source position from TDOA within three pairs of recorders. Uses the
    Gauss-Newton iterative algorithm
- localise_allpairs
    Find source position from TDOA witin more than three pairs of recorders.
    Runs the Gauss-Newton algorithm for all triplets of recorders pairs for
    cross-validation

TDOA computation
----------------
- cross_correlation
    Measures TDOA through a simple cross_correlation method
- gcc_phat
    Measures TDOA using the GCC-PHAT method with relies on phase information

Visualisation tools
-------------------
- plot_hyperboloids_solutions
    Visualises recorder position, revolution hyperboloids for each pair of
    recorders, and estimated source positions in a single 3D plot.

Utilitary functions
-------------------
- custom_butterworth_filter
    Defines a custom butterworth bandpass filter. Uses a dual pass method
    to avoid phase shift before TDOA computation
- tdoa_to_distancediff
    Translates time diffence of arrival into intra-pair difference in distance
    to the sound source
- cost_function
    Measures the consistency between current source position estimation and
    known intra-pair distance to source differences. Used in the iterative
    algorithm
- plot_hyperboloid
    Visualises the revolution hyperboloid which constrains source position from
    the TDOA of a single pair of recorders in a 3D plot
"""

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift
from scipy.optimize import least_squares
from itertools import combinations

def custom_butterworth_filter(signal, fs, fmin=250, fmax=5000, order=2):
    """
    Defines a custom Butterworth filter and filters the input data through
    a forward-backward pass to avoid phase shift.
    
    Arguments
    ---------
    signal (np.array): the signal to filter. If multidimensional, filter along
        the first axis.
    fs (int): the sampling frequency of the signal
    fmin (float): the lower bound of the bandpass filter
    fmax (float): the upper bound of the bandpass filter
    order (int): the order of the filter. Since the function makes a double
        pass, the final order will be order * 2.
    
    Returns
    -------
    signal_filt (np.array): the filtered signal. 
    """
    
    filter = sig.butter(order, (fmin, fmax), btype='bandpass',
                        output='sos', fs=fs
                        )
    signal_filt = sig.sosfiltfilt(filter, signal, axis=0)
    
    return signal_filt


def cross_correlation(signal1, signal2, fs,  plot=False,
                      filter=True, **filterargs
                      ):
    """
    Measure the time difference of arrival of a sound between two recorders.
    Filter the audio and compute cross-correlation to determine tdoa.
    
    Arguments
    ---------
    signal1 (np.array): the recording from the first recorder
    signal2 (np.array): the recording from the second recorder
    fs (int): sampling frequency of the recorders. Must be the same for both
        signals
    plot (bool): whether to display cross_correlation
    filter (bool): whether to apply preprocessing or not
        See custom_butterworth_filter for details
    
    Returns
    -------
    tdoa (float): time difference of arrival between both recorders
    Diplays a plot of cross-correlation if plot is True
    """
    
    # Preprocessing
    if filter:
        signal1 = custom_butterworth_filter(signal1, fs, **filterargs)
        signal2 = custom_butterworth_filter(signal2, fs, **filterargs)
    
    # Measure time difference of arrival from cross-correlation
    cor = sig.correlate(signal1, signal2)
    lags = sig.correlation_lags(len(signal1), len(signal2)) / fs
    tdoa = lags[np.argmax(cor)]
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(lags, cor)
        ax.set_ylabel('Correlation')
        ax.set_xlabel('Lag (s)')
        plt.show()
    
    return tdoa


def gcc_phat(signal1, signal2, fs, max_delay=0.015,
             plot=False, filter=True, **filterargs):
    """
    Measures the time difference of arrival of one sound at two recorder using
    the GCC-PHAT method.
    
    Arguments
    ---------
    signal1 (np.array): the recording at recorder 1
    signal2 (np.array): the recording at recorder 2 
    fs (int): sampling frequency of the recorders. Must be the same
    max_delay (float): maximum time delay measurable, depends on the recorders
        relative positions.
    plot (bool): whether to display the GCC-PHAT results
    filter (bool): whether to filter the signal before computing GCC-PHAT
    filterargs (dict): the parameters of the filter.
        See custom_butterworth_filter for details
    
    Returns
    -------
    tdoa (float): the time difference of arrival between recorders
    Displays the GCC-PHAT is plot is True
    
    Note
    ----
    Filtering may results in a phase shift of the signal and an artifact at
    the edges of the recordings. custom_butterworth_filter uses a dual-pass
    method to prevent phase shift, and the function discards 100 ms at the
    beginning and end of the filtered signals to remove the artifacts.
    """
    
    if filter:
        signal1 = custom_butterworth_filter(signal1, fs, **filterargs)
        signal2 = custom_butterworth_filter(signal2, fs, **filterargs)
        # Remove 100 ms at start and end of the sounds to remove artifacts
        signal1 = signal1[int(0.3 * fs): int(len(signal1) - (0.3 *fs))]
        signal2 = signal2[int(0.3 * fs): int(len(signal2) - (0.3 *fs))]
        # Apply Hann window to the trimmed signals
        window = sig.get_window('hann', len(signal1))
        signal1 *= window
        signal2 *= window
    
    # Fourier Transform uses sum of length to avoid aliasing
    fft_length = len(signal1) + len(signal2)
    spectrum1 = fft(signal1, n=fft_length)
    spectrum2 = fft(signal2, n=fft_length)
    
    # Compute cross-spectrum and normalise by magnitude
    R = spectrum1 * np.conj(spectrum2)
    R /= np.abs(R) + 1e-10  # Add small number to avoid dividing by 0
    
    # Inverse Fourier transform
    corr = np.real(ifft(R, n=fft_length))
    corr = fftshift(corr)
    
    # Limit search to physically possible delays
    max_delay_samples = max_delay * fs
    low_bound = int(fft_length // 2 - max_delay_samples)
    high_bound = int(fft_length // 2 + max_delay_samples)
    corr_window = corr[low_bound: high_bound]    
    
    # Get peak relative to middle (= true lag)
    peak_idx = np.argmax(np.abs(corr_window)) + low_bound
    
    # Translate peak_idx into actual tdoa
    lags = (np.arange(fft_length) - fft_length // 2) / fs
    tdoa = lags[peak_idx]
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(lags, corr)
        ax.set_ylabel('GCC-PHAT')
        ax.set_xlabel('Lag (s)')
        plt.show()
    
    return tdoa


def tdoa_to_distancediff(tdoa, sound_speed=343):
    """
    Translates a time difference of arrival into a difference in distance 
    between the recorders and the sound source.
    
    Arguments
    ---------
    tdoa (float): the time difference of arrival
    sound_speed (float): the speed of sound
    
    Returns
    -------
    distance_diff (float): the difference in distance to sound sources
    """
    
    return tdoa * sound_speed


def cost_function(mic_positions, true_distance_diffs, candidate_pos):
    """
    Cost function of the iterative localiser.
    Computes the difference in relative distance to recorder pairs between
    a candidate position and ground truth data based on TDOA.
    
    Arguments
    ---------
    mic_positions (np.array of size (2*k, 3): positions of the recorders
    true_distance_diffs (np.array of size k): difference in distance to source
        within each recorder pair
    candidate_pos (np.array of size 3): the candidate source position to
        evaluate
    Returns
    -------
    cost (float): value of the cost function
    """
    
    cost = 0
    for k in range(len(true_distance_diffs)):  # For each recorder pair
        d1 = np.linalg.norm(candidate_pos - mic_positions[2 * k, :])
        d2 = np.linalg.norm(candidate_pos - mic_positions[2 * k + 1, :])
        cost += ((d1 - d2) - true_distance_diffs[k]) ** 2
    
    return cost


def localise_3pairs(mic_positions, true_distance_diffs,
                    initial_guess=None, bounds=None
                    ):
    """
    Find the sound position from the recorder positions, the relative distance
    to source differences within each recorder pair, using the Gauss-Newton
    algorithm.
    
    Arguments
    ---------
    mic_positions (np.array of size (6, 3): the positions of the recorders
    true_distance_diffs (np.array of size 3): the relative distance to the
        sound source within each pair of recorders
    initial_guess (np.array of size 3): the initial estimation
    bounds (np.array of size 3): physical limits for the source position
    
    Returns
    -------
    source_position (np.array of size 3): the position of the sound source
    info (dict): annex information about the procedure.
        Includes keys 'residual', 'nb_iter'
    """
    
    if initial_guess is None:
        initial_guess = np.mean(mic_positions, axis=0)
    
    if bounds is None:  # Loose bounds based on mic_positions
        mins = np.min(mic_positions, axis=0) - 1
        maxs = np.max(mic_positions, axis=0) + 1
        bounds = np.array(list(zip(mins, maxs)))
    
    # Wrap cost function to turn it into a single arg function
    # Needed for least_squares
    def objective(pos):
        res = cost_function(mic_positions, true_distance_diffs, pos)
        return res
    
    # Minimisation avec contraintes de bornes
    results = least_squares(fun=objective,
                            x0 = initial_guess,
                            jac='3-point',
                            bounds=(bounds[:, 0], bounds[:, 1]),
                            method='trf',  # or method='L-BFGS-B',
                            ftol=1e-12,
                            xtol=1e-8,
                            gtol=1e-8,
                            max_nfev=1000
                            )
       
    # Calcul des diagnostics
    source_position = results.x
    info = {'success': results.success,
            'nb_iter': results.nfev,
            'final_cost': results.fun,
            'message': results.message
            }
    
    return source_position, info


def localise_allpairs(mic_positions, signals, sr,
                      tdoa_func=gcc_phat, filter=True,
                      bounds=None, initial_guess=None, **filterargs
                      ):
    """
    Compute sound source position from recorder position and recorded signals.
    
    Arguments
    ---------
    mic_positions (np.array of size (2*k, 3)): the position of the recorders
    signals(np.array of size (n, 2*k)): the recordings of each recorder
    sr (int): the sampling frequency of the recorders
    tdoa_func (function): how to measure tdoa. Default is gcc_phat
    filter (bool): whether to filter the signals before TDOA estimation
    **filterargs: additional arguments for the filter function
    **args: additional arguments for scipy.least_squares
    bounds (np.array): the physical bounds of the sound source position
    initial_guess np.array of size 3): initial position for the iterative
        algorithm
    
    Returns
    -------
    estimates (list): list of possible sound source position
    info (dict): additional information about solution quality
    """
    
    n_pairs = mic_positions.shape[0] // 2
    
    # Measure TDOA
    tdoas = []
    for k in range(n_pairs):
        signal1 = signals[:, 2 * k]
        signal2 = signals[:, 2 * k + 1]
        tdoa = tdoa_func(signal1, signal2, sr, filter=filter, **filterargs)
        tdoas.append(tdoa)
    
    # Get distance to source difference within each recorder pair
    true_distance_diffs = [tdoa_to_distancediff(elt) for elt in tdoas]
    
    # Solve position for all triplets of recorder pairs
    estimates = []
    info_dict = {'success': [],
                 'nb_iter': [],
                 'final_cost': [],
                 'message': []
                 }
    for a, b, c in combinations(range(n_pairs), 3):
        triplet_positions = mic_positions[[2 * a, 2 * a + 1,
                                           2 * b, 2 * b + 1,
                                           2 * c, 2 * c + 1
                                           ]
                                          ]
        triplet_distdiffs = [true_distance_diffs[i] for i in [a, b, c]]
        source_pos, info = localise_3pairs(triplet_positions,
                                           triplet_distdiffs,
                                           initial_guess=initial_guess,
                                           bounds=bounds
                                           )
        estimates.append(source_pos)
        info_dict['success'].append(info['success'])
        info_dict['nb_iter'].append(info['nb_iter'])
        info_dict['final_cost'].append(info['final_cost'])
        info_dict['message'].append(info['message'])
    
    centroid = np.mean(np.array(estimates), axis=0)
    spread = np.std(np.array(estimates), axis=0)
    max_spread = np.max(np.linalg.norm(np.array(estimates) - centroid, axis=1))
    
    print(f"Solution centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}) m")
    print(f"Solution std: [{spread[0]:.2f}, {spread[1]:.2f}, {spread[2]:.2f}] m")
    print(f"Max spread: {max_spread:.2f} m")
        
    if max_spread > 1.0:
        print("High spread: CHECK DATA")
    
    return estimates, info_dict


def plot_hyperboloid(mic_positions, distance_diff, n_step=50, col='Blue',
                     xyz_lims=np.array([[0, 26], [0, 8], [0, 3.5]]),
                     fig=None, ax=None
                     ):
    """
    Plots the hyperboloid resulting from the recorder positions and the
    distance difference from the source to each recorder.
    Uses polar coordinates around the axis of the microphone pair.
    
    Arguments
    ---------
    mic_positions (np.array): the (x, y, z) positions of both recorders
    distance_diff (float): the difference in distance to the sound source
        between recorder 2 and recorder 1
    n_step (int): the number of points in the grid
    fig (plt.Figure): an existing figure to plot in. If None, makes a new fig
    ax (plt.Axes): an existing artist. Must have projection='3d'. If None,
        makes a new artist.
    
    Returns
    -------
    fig, ax (plt.Figure, plt.Axes): the figure with the plotted hyperboloid.
    """
    
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(projection='3d')
    
    # Define reference axis
    axis_length = np.linalg.norm(mic_positions[1] - mic_positions[0])
    axis_direction = (mic_positions[1] - mic_positions[0]) / axis_length
    axis_center = (mic_positions[0] + mic_positions[1]) / 2
    focal_dist = axis_length / 2
    
    if distance_diff >= axis_length:
        print("Distance difference physically impossible")
        return
    
    # Set parameters
    a = distance_diff / 2
    b = np.sqrt((axis_length / 2)**2 - a**2)
     
    # Make polar grid
    u = np.linspace(-np.pi, np.pi, n_step)
    v = np.linspace(-5, 5, n_step)  # Change to accommodate aviary dimensions
    
    # Create revolution hyperboloid
    X = np.zeros(n_step**2, dtype=float)
    Y = np.zeros(n_step**2, dtype=float)
    Z = np.zeros(n_step**2, dtype=float)
    
    # Prepare rotation matrix
    up = np.array([0, 0, 1])
    vec_x = np.cross(up, axis_direction)  # normal to up and axis_dir
    vec_x = vec_x / np.linalg.norm(vec_x)
    vec_y = np.cross(axis_direction, vec_x)
    
    for i, vi in enumerate(v):
        for j, uj in enumerate(u):
            # Coordinates in polar coordinates
            x_polar = a * np.cosh(vi)
            y_polar = b * np.sinh(vi) * np.cos(uj)
            z_polar = b * np.sinh(vi) * np.sin(uj)
            
            
            # Run rotation and add point to point list
            xyz_cartesian = (axis_center +
                             x_polar * axis_direction +  # x-axis = axis_dir
                             y_polar * vec_x +
                             z_polar * vec_y
                             )
            X[i * n_step + j] = xyz_cartesian[0]
            Y[i * n_step + j] = xyz_cartesian[1]
            Z[i * n_step + j] = xyz_cartesian[2]
    
    # TODO
    # Make different colours to allow the visualisation of several hyperboloids
    # Make sure alpha ~= 0.25 so that all hypoerboloids are visible
    
    # Only show points inside the volume of interest
    points_to_plot = np.array([(x, y, z) for (x, y, z) in zip(X, Y, Z)
                               if (xyz_lims[0, 0] < x < xyz_lims[0, 1] and
                                   xyz_lims[1, 0] < y < xyz_lims[1, 1] and
                                   xyz_lims[2, 0] < z < xyz_lims[2, 1]
                                   )
                              ]
                              )
    
    ax.scatter(points_to_plot[:, 0],
               points_to_plot[:, 1],
               points_to_plot[:, 2],
               color=col, alpha=0.25
               )
    
    # Show microphone positions
    ax.scatter(mic_positions[:, 0], mic_positions[:, 1], mic_positions[:, 2],
               color=col, s=100, marker='o'
               )
    
    # Restrict to volume of interest
    ax.set_xlim((xyz_lims[0, 0], xyz_lims[0, 1]))
    ax.set_ylim((xyz_lims[1, 0], xyz_lims[1, 1]))
    ax.set_zlim((xyz_lims[2, 0], xyz_lims[2, 1]))
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    plt.tight_layout()
    
    return fig, ax


def plot_hyperboloids_solutions(source_positions, mic_positions,
                                distance_diffs, n_step=50,
                                bounds=np.array([[0, 26], [0, 8], [0, 3.5]]),
                                colours=None, fig=None, ax=None
                                ):
    """
    Makes a 3D plot of the volume of interest with the recorder positions,
    the revolution hyperboloids defined by intra-pair distance to source diffs
    and the estimated source positions
    
    Arguments
    ---------
    source_positions (np.array): the estimated positions of the sound source
    mic_positions (np.array): the positions of the recorders
    distance_diffs (list): the intra-pair differences in distance to source
    n_step (int): number of points for the hyperboloid rendering
    bounds (np.array): the bounds of the volume of interest
    colours (list): colour for each hyperboloid
    
    Returns
    -------
    fig, ax: the figure and artist with the plot
    """
    
    if fig is None:
        fig = plt.figure(figsize=(12, 8))
    if ax is None:
        ax = fig.add_subplot(projection='3d')
    
    if colours is None:
        colours = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    
    n_pairs = len(mic_positions) // 2
    
    # Plot hyperboloids and recording positions
    for k in range(n_pairs):
        plot_hyperboloid(mic_positions[[2 * k, 2 * k + 1]], distance_diffs[k],
                         n_step=n_step, col=colours[k], xyz_lims=bounds,
                         fig=fig, ax=ax
                         )
    
    # Plot source positions
    ax.scatter([a[0] for a in source_positions],
               [a[1] for a in source_positions],
               [a[2] for a in source_positions],
               c='black', marker="*", s=100
               )
    
    return fig, ax
