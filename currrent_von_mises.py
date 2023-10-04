import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0
from scipy.stats import vonmises

def von_mises_tuning(directions, preferred_direction, concentration):
    """
    Generate a von Mises tuning curve.

    Args:
        directions (array): Array of stimulus directions (in radians).
        preferred_direction (float): Preferred direction of the neuron (in radians).
        concentration (float): Concentration parameter (controls tuning width).

    Returns:
        tuning_curve (array): Probability of firing for each stimulus direction.
    """
    directions = 2*(directions-preferred_direction)
    tuning_curve = np.exp(concentration * np.cos(directions - preferred_direction))
    tuning_curve /= (2 * np.pi * i0(concentration))  # Normalize

    return tuning_curve

# Define parameters
preferred_direction = np.pi / 4  # Preferred direction (e.g., 45 degrees)
concentration = 5.0  # Concentration parameter (controls tuning width)

# Generate stimulus directions from 0 to 2*pi radians
directions = np.linspace(-np.pi/2, np.pi/2, 100)

# Generate the tuning curve
tuning_curve = von_mises_tuning(directions, preferred_direction, concentration)

# Plot the tuning curve
plt.figure()
plt.plot(directions, tuning_curve)
plt.xlabel("Stimulus Direction (radians)")
plt.ylabel("Response Probability")
plt.title("von Mises Tuning Curve")

I_max = 1

I_stim = I_max*tuning_curve/np.max(tuning_curve)

# Plot the tuning curve
plt.figure()
plt.plot(directions, I_stim)
plt.xlabel("Stimulus Direction (radians)")
plt.ylabel("Current")
plt.title("von Mises Tuning Curve")



loc = np.pi/4  # circular mean
kappa = 5  # concentration

deneme = vonmises.pdf(2*directions, kappa, 2*preferred_direction)
I_stim = I_max * deneme / np.max(deneme)
print('Done')

plt.figure()
plt.plot(directions, deneme)
plt.xlabel("Stimulus Direction (radians)")
plt.ylabel("Current")
plt.title("von Mises Tuning Curve")

plt.figure()
plt.plot(directions, I_stim)
plt.xlabel("Stimulus Direction (radians)")
plt.ylabel("Current")
plt.title("von Mises Tuning Curve")




plt.show()
plt.close()