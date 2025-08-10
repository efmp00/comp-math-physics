# Libraries
import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors

# Plot parameters
plt.rcParams.update({
    'lines.linewidth': 2,               # linewidth
    'font.family': 'Helvetica Neue',    # font
    'mathtext.fontset': 'cm',           # math font
    'mathtext.default': 'it',           # math font style
    'font.size': 24,                    # font size
    'axes.titlesize': 24,               # title size
    'axes.grid': True,                  # grid
    'grid.linestyle': '-.',             # grid style
    'axes.facecolor': '#ECECEC',        # background color for the axes
    'figure.facecolor': '#FFFFFF',      # background color for the axes
    'legend.facecolor': '#FFFFFF'       # background color for the legend
})

def gauss_propagation(wavelength, grid_points, focal_length = None):
    # Physical parameters
    w0 = 5e-4                       # Beam waist radius [m]
    k = 2 * np.pi / wavelength      # Wavenumber
    zR = (w0 ** 2 * k) / 2          # Rayleigh range
    z_max = 1.5 * zR                # Maximum propagation distance

    # Numerical grid
    N = grid_points                 # Number of grid points
    window_size = 3 * w0            # Window size
    dx = 2 * window_size / N        # Spatial step
    max_spatial_freq = np.pi / dx   # Maximum spatial frequency

    # Coordinates
    grid_normalized = np.arange(-N / 2, N / 2) * (2 / N)
    x = grid_normalized * window_size
    y = grid_normalized * window_size
    z = np.linspace(0, z_max, N)

    # Frequency domain
    kx = grid_normalized * max_spatial_freq
    ky = grid_normalized * max_spatial_freq

    # Meshgrids
    X, Y = np.meshgrid(x, y)
    Kx, Ky = np.meshgrid(kx, ky)

    # Initial field (Gauss)
    r = X ** 2 + Y ** 2
    GB0 = np.exp(-r / w0 ** 2)

    # Propagation
    GB0_fft = np.fft.fftshift(np.fft.fft2(GB0))
    GB_prop = np.zeros((N, len(z)), dtype = complex)

    for n, zn in enumerate(z):
        propagator = np.exp(-1j * zn / (2 * k) * (Kx ** 2 + Ky ** 2))
        GBz = np.fft.ifft2(GB0_fft * propagator) * np.exp(1j * k * zn)
        GB_prop[:, n] = GBz[N // 2, :]

    Z_mesh, Y_mesh = np.meshgrid(z, y)
    intensity = np.abs(GB_prop) ** 2

    if focal_length is None:
        return X, Y, GB0, Z_mesh, Y_mesh, intensity
    else:
        f = focal_length * zR
        thin_lens = np.exp(-1j * k * r / (2 * f))

        # Initial field (with lens)
        GB0_lens = GB0 * thin_lens
        
        # Propagation
        GB0_lens_fft = np.fft.fftshift(np.fft.fft2(GB0_lens))
        GB_lens_prop = np.zeros((N, len(z)), dtype = complex)

        for n, zn in enumerate(z):
            propagator = np.exp(-1j * zn / (2 * k) * (Kx ** 2 + Ky ** 2))
            GBz = np.fft.ifft2(GB0_lens_fft * propagator) * np.exp(1j * k * zn)
            GB_lens_prop[:, n] = GBz[N // 2, :]
        intensity_GB_lens = np.abs(GB_lens_prop) ** 2

        return X, Y, GB0, Z_mesh, Y_mesh, intensity, GB0_lens, intensity_GB_lens

def plot_initial_condition(X, Y, field, title):
    plt.figure(figsize = (16, 9))
    plt.pcolormesh(X, Y, np.abs(field), shading = 'auto', cmap = 'jet')
    plt.xlabel(r"$x \: [\mathrm{m}]$")
    plt.ylabel(r"$y \: [\mathrm{m}]$")
    plt.ticklabel_format(style = 'scientific', axis = 'both', scilimits = (0, 0))
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def plot_propagation(Z, Y, intensity, title):
    plt.figure(figsize = (16, 9))
    plt.pcolormesh(Z, Y, intensity, shading = 'auto', cmap = 'jet')
    plt.xlabel(r"$z \: [\mathrm{m}]$")
    plt.ylabel(r"$y \: [\mathrm{m}]$")
    plt.ticklabel_format(style = 'scientific', axis = 'both', scilimits = (0, 0))
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

X, Y, GB0, Z_mesh, Y_mesh, intensity, GB0_lens, intensity_GB_lens = gauss_propagation(500e-9, 2 ** 8, 0.5)

plot_initial_condition(
    X, Y, GB0,
    rf'Initial Gaussian Beam'
    )

plot_propagation(
    Z_mesh, Y_mesh, intensity,
    rf'Gaussian Beam free-space propagation'
    )

plot_initial_condition(
    X, Y, GB0_lens,
    rf'Initial Gaussian Beam through a thin lens'
    )

plot_propagation(
    Z_mesh, Y_mesh, intensity_GB_lens,
    rf'Gaussian Beam propagation through a thin lens ($f = 0.5 z_{{\mathrm{{R}}}}$)'
    )

# === Bessel-Gauss beam ===
def bessel_gauss_propagation(w0, wavelength = 632.8e-9, m = 0, N = 2 ** 8, z_factor = 1.5):
    # Physical parameters
    k = 2 * np.pi / wavelength          # Wavenumber
    kt = 8665                           # Transveral wavenumber
    zR = (w0 ** 2 * k) / 2              # Rayleigh range
    z_max = z_factor * zR               # Propagation distance
    
    # Numerical grid
    window_size = 2 * w0                # Simulation window
    dx = 2 * window_size / N            # Spatial step
    max_spatial_freq = np.pi / dx       # Max spatial frequency
    
    # Coordinates
    grid_normalized = np.arange(-N / 2, N / 2) * (2 / N)
    x = grid_normalized * window_size
    y = grid_normalized * window_size
    z = np.linspace(0, z_max, N)
    
    # Frequency domain
    kx = grid_normalized * max_spatial_freq
    ky = grid_normalized * max_spatial_freq

    # Meshgrids
    X, Y = np.meshgrid(x, y)
    Kx, Ky = np.meshgrid(kx, ky)
    
    # Initial field (Bessel-Gauss)
    r = np.sqrt(X ** 2 + Y ** 2)
    Bessel = sc.jv(m, kt * r)
    BGB0 = np.exp(-r ** 2 / w0 ** 2) * Bessel * np.exp(1j * m * np.arctan2(Y, X))
    
    # Propagation
    BGB0_fft = np.fft.fftshift(np.fft.fft2(BGB0))
    BGB_prop = np.zeros((N, len(z)), dtype = complex)
    
    for n, zn in enumerate(z):
        propagator = np.exp(-1j * zn / (2 * k) * (Kx**2 + Ky**2))
        BGBz = np.fft.ifft2(BGB0_fft * propagator) * np.exp(1j * k * zn)
        BGB_prop[:, n] = BGBz[N // 2, :]
    
    # Prepare for plotting
    Z_mesh, Y_mesh = np.meshgrid(z, y)
    intensity = np.abs(BGB_prop) ** 2
    
    return X, Y, BGB0, Z_mesh, Y_mesh, intensity

# Waist values
waists = [0.5e-3, 1.5e-3, 3.0e-3]  # Beam waists to test

for w0 in waists:
    X, Y, BGB0, Z_mesh, Y_mesh, intensity = bessel_gauss_propagation(w0)
    
    plot_initial_condition(
        X, Y, BGB0,
        rf'Initial Bessel-Gauss Beam ($w_{0} = {w0 * 1e3:.1f} \: \mathrm{{mm}}$)'
    )
    
    plot_propagation(
        Z_mesh, Y_mesh, intensity,
        rf'Bessel-Gauss Beam propagation ($w_{0} = {w0 * 1e3:.1f} \: \mathrm{{mm}}$)'
    )
