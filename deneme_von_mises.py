# Import necessary libraries
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from network_functions import NetworkAnalyzer
from plotting_functions import PlottingFuncs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
from scipy.signal import convolve2d
from scipy.signal import convolve
from scipy.special import i0
from scipy.stats import vonmises
import networkx as nx
import nest
import nest.raster_plot


# Simulation time in ms
simtime = 30000.0  

# number of excitatory and inhibitory neurons
order = 400
NE = order  
NI = order  

# Define Simulation Parameters
bin_width = 200.0 # bin width the calculate firing rate
delay = 1.5  # synaptic delay in ms
eta = 2.0  # external rate relative to threshold rate
epsilon = 0.1  # connection probability

N_neurons = NE + NI  # number of neurons in total
CE = int(epsilon * NE)  # number of excitatory synapses per neuron
CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
C_tot = int(CI + CE)  # total number of synapses per neuron

# Define Neuron Parameters
tauMem = 20.0  # time constant of membrane potential in ms
theta = 20.0  # membrane threshold potential in mV
neuron_params = {"C_m": 1.0,
                 "tau_m": tauMem,
                 "t_ref": 2.0,
                 "E_L": 0.0,
                 "V_reset": 0.0,
                 "V_m": 0.0,
                 "V_th": theta}
J = 1  # postsynaptic amplitude in mV
J_ex = J  # amplitude of excitatory postsynaptic potential
  
connection_seed=70

preferred_direction = 0*np.pi / 4  # Preferred direction (e.g., 45 degrees), one can change the value whatever value they want

# Generate stimulus directions from -pi/2 to pi/2 radians
pyrng_gen = np.random.RandomState(connection_seed)

# Creation of orientation preferences
pref_angles_p_exc = np.linspace(-np.pi/2, np.pi/2, NE)
pref_angles_p_inh = np.linspace(-np.pi/2, np.pi/2, NI)

# Seed to get same shuffling
np.random.seed(13)
# Shuffling orientation preferences
np.random.shuffle(pref_angles_p_exc)
np.random.shuffle(pref_angles_p_inh)


#kappa value to create input currents for each input according to visual stimulus for von mises distribution
kappa_amp_stim=10

#kappa value to create weights between neurons  for von mises distribution
kappa_conn_weight = 10

# a is E->I dominance, g is inhibition dominance value. One can change them to look at inhibition dominance effect during perturbation
a=2
g=-3

J_in = g * J_ex # amplitude of inhibitory postsynaptic potential

# Creation of 2D weight array
ww_EE = np.zeros((NE, NE))
ww_EI = np.zeros((NE, NI))
ww_IE = np.zeros((NI, NE))
ww_II = np.zeros((NI, NI))

# Value to scale weigths
p = 0.1

# In each for loop, weights are created for each neuron using von mises distribution
for i in range(NE):
    ww_EE[:, i] = p*vonmises.pdf(2*pref_angles_p_exc, kappa_conn_weight, 2*pref_angles_p_exc[i])

for i in range(NI):
    ww_EI[:, i] = a*p*vonmises.pdf(2*pref_angles_p_exc, kappa_conn_weight, 2*pref_angles_p_inh[i])

for i in range(NE):
    ww_IE[:, i] = g*p*vonmises.pdf(2*pref_angles_p_inh, kappa_conn_weight, 2*pref_angles_p_exc[i])

for i in range(NI):
    ww_II[:, i] = g*p*vonmises.pdf(2*pref_angles_p_inh, kappa_conn_weight, 2*pref_angles_p_inh[i])


# Helper Functions
analyzer = NetworkAnalyzer(NE, NI, N_neurons, simtime, bin_width)
m_plot = PlottingFuncs(N_neurons, simtime, bin_width, CE, CI)


# One can run many simulations using this function
def run_sim(random_seed, plotting_flag, sim):

    # Reset previous simulations
    nest.ResetKernel()
    # Set the number of threads you want to use
    num_threads = 2
    # Set the kernel status to change the numbers of threads
    nest.SetKernelStatus({"local_num_threads": num_threads})
    # Set connection seed
    nest.SetKernelStatus({"rng_seed": connection_seed})
    dt = 0.1  # the resolution in ms
    nest.resolution = dt
    nest.print_time = True
    nest.overwrite_files = True

    # Poisson rate values for inhibitory and excitatory neuron populations. Poisson is used to get a balanced network 
    rates_exc = 1000
    rates_inh = 1000


    print("Building network")
    # The network is all-to-all connected. Autapses (self connections) were set to false.
    conn_params_ex = {"rule": "all_to_all", "allow_autapses": False}
    conn_params_in = {"rule": "all_to_all", "allow_autapses": False}

    #Define the Positions
    pos_ex = nest.spatial.free(pos=nest.random.uniform(min=-5.0, max=5.0), num_dimensions=3)

    # Create excitatory neurons, inhibitory neurons, poisson spike generator, and spike recorders
    nodes_ex = nest.Create("iaf_psc_delta", NE, params=neuron_params, positions=pos_ex)
    nodes_in = nest.Create("iaf_psc_delta", NI, params=neuron_params, positions=pos_ex)
    espikes = nest.Create("spike_recorder", params={"start": 3000.0, "stop":simtime})
    ispikes = nest.Create("spike_recorder", params={"start": 3000.0, "stop":simtime})

    #Define the Synapses
    nest.CopyModel("static_synapse", "excitatory", {"weight": J_ex, "delay": delay})
    nest.CopyModel("static_synapse", "inhibitory", {"weight": J_in, "delay": delay})
    

    # Create Connections between populations
    nest.Connect(nodes_ex, nodes_ex, conn_params_ex, syn_spec={"weight": ww_EE.T, "delay": delay})
    nest.Connect(nodes_ex, nodes_in, conn_params_ex, syn_spec={"weight": ww_EI.T, "delay": delay})
    nest.Connect(nodes_in, nodes_ex, conn_params_in, syn_spec={"weight": ww_IE.T, "delay": delay})
    nest.Connect(nodes_in, nodes_in, conn_params_in, syn_spec={"weight": ww_II.T, "delay": delay})

    nest.SetKernelStatus({"rng_seed": random_seed})
    # Connect the poisson generator to each neuron
    noise_exc = nest.Create("poisson_generator", params={"rate": rates_exc})
    noise_inh = nest.Create("poisson_generator", params={"rate": rates_inh})

    # Connect Noise Generators 
    nest.Connect(noise_exc, nodes_ex)
    nest.Connect(noise_inh, nodes_in)

    # Connect recorders to record activity
    nest.Connect(nodes_ex, espikes)
    nest.Connect(nodes_in, ispikes)


    # One can set here how many neurons to perturb
    # The perturbed one will have close orientation preference to preferred direction
    num_of_perturbed_neuron = 1
    abs_array = abs(preferred_direction-pref_angles_p_exc)
    sorted_indices = np.argsort(abs_array)
    src_indices = sorted_indices[0:num_of_perturbed_neuron]

    abs_array_inh = abs(preferred_direction-pref_angles_p_inh)
    sorted_indices_inh = np.argsort(abs_array_inh)
    src_indices_inh = sorted_indices_inh[0:num_of_perturbed_neuron]

    # One can change the base amplitude to model different visual stimulus contrasts
    base_amplitude=5
    
    # src numbers start from 1. Hence, an incrementation is applied.
    src_id = src_indices+1
    src_id_inh = src_indices_inh+1

    # t is to scale input current for visual stimulus.
    # IF ONE WANTS TO PRESENT VISUAL STIMULUS DURING PERTURBATION, THE FOLLOWING LINES SHOULD BE UNCOMMENTED
    '''
    t=0.1
    amplitude_exc= vonmises.pdf(2*pref_angles_p_exc, kappa_amp_stim, 2*pref_angles_p_exc[src_indices[0]])
    amplitude_exc = t*amplitude_exc*base_amplitude/np.max(amplitude_exc)

    amplitude_inh= vonmises.pdf(2*pref_angles_p_inh, kappa_amp_stim, 2*pref_angles_p_inh[src_indices_inh[0]])
    amplitude_inh = t*amplitude_inh*base_amplitude/np.max(amplitude_inh)

    # Connect the stimulator to the neuron
    for i in range(NE):
        stim_params_exc = {"amplitude": amplitude_exc[i], "start": 4000.0, "stop": simtime}
        stimulator_exc = nest.Create("dc_generator", params=stim_params_exc)
        nest.Connect(stimulator_exc, nodes_ex[i])

    for i in range(NI):        
        stim_params_inh = {"amplitude": amplitude_inh[i], "start": 4000.0, "stop": simtime}
        stimulator_inh = nest.Create("dc_generator", params=stim_params_inh)
        nest.Connect(stimulator_inh, nodes_in[i])
    '''
    
    # Parameters for perturbation
    stim_params_1 = {"amplitude": 160.0, "start": 4000.0, "stop": simtime}
    stim_1= nest.Create("dc_generator", params=stim_params_1)

    # The neuron is perturbed if sim parameter is True
    if sim==True:                            
        for k in range(len(src_id)):
            print(src_id)
            nest.Connect(stim_1, nodes_ex[src_indices[k]])
            # One can also perturb an inhibitory neuron
            #nest.Connect(stim_1, nodes_in[src_indices_inh[k]])

    # Start Simulation
    print("Simulating")

    nest.Simulate(simtime)

    # Extract Some Parameters from the Simulation
    events_ex = espikes.n_events
    events_in = ispikes.n_events

    rate_ex = events_ex / simtime * 1000.0 / NE
    rate_in = events_in / simtime * 1000.0 / NI

    num_synapses_ex = nest.GetDefaults("excitatory")["num_connections"]
    num_synapses_in = nest.GetDefaults("inhibitory")["num_connections"]
    num_synapses = num_synapses_ex + num_synapses_in

    # Extract spikes and plot raster plot
    sr1_spikes = espikes.events['senders']
    sr1_times = espikes.events['times']
    sr2_spikes = ispikes.events['senders']
    sr2_times = ispikes.events['times']

    # Calculate avg. firing rates of excitatory neurons
    avg_firing_exc, spike_times_exc = analyzer.calculate_avg_firing(nodes_ex, simtime, sr1_spikes, sr1_times, 0)

    # Calculate avg. firing rates of inhibitory neurons
    avg_firing_inh, spike_times_inh = analyzer.calculate_avg_firing(nodes_in, simtime, sr2_spikes, sr2_times, 1)


    # Calculate CoV of excitatory neurons
    CoV_exc = analyzer.calculate_CoV(spike_times_exc)
    # Calculate CoV of inhibitory neurons
    CoV_inh = analyzer.calculate_CoV(spike_times_inh)


    # Calculating firing rates for both populations
    hist_counts_all_exc, bin_centers_exc, avg_hist_counts_exc = analyzer.calculating_firing_rates(src_id, spike_times_exc, 0)

    hist_counts_all_inh, bin_centers_inh, avg_hist_counts_inh = analyzer.calculating_firing_rates(src_id_inh, spike_times_inh, 1)


    hist_counts_all_exc_mean = np.mean(hist_counts_all_exc, axis=1)
    hist_counts_all_exc_mean = np.squeeze(hist_counts_all_exc_mean)
    hist_counts_all_inh_mean = np.mean(hist_counts_all_inh, axis=1)
    hist_counts_all_inh_mean = np.squeeze(hist_counts_all_inh_mean)
    
    # To delete the activity of perturbed excitatory neuron
    diff_exc = pref_angles_p_exc
    diff_new_exc = np.delete(diff_exc, src_indices)
    diff_new_exc_indices = np.argsort(diff_new_exc)

    # To delete the activity of perturbed inhibitory neuron
    diff_inh = pref_angles_p_inh
    diff_new_inh = np.delete(diff_inh, src_indices_inh)
    diff_new_inh_indices = np.argsort(diff_new_inh)

    #return the required values for simulation
    return nodes_ex, nodes_in, bin_centers_exc, avg_hist_counts_exc, avg_hist_counts_inh, spike_times_exc, spike_times_inh, \
        hist_counts_all_exc_mean[diff_new_exc_indices], diff_new_exc[diff_new_exc_indices], hist_counts_all_inh_mean[diff_new_inh_indices], diff_new_inh[diff_new_inh_indices]




plotting_flag = False

# Define the number of runs. 10 simulations will be with perturbation, 10 without perturbation
num_runs = 20

# Lists to store the results
nodes_ex = []
nodes_in = []
hist_mean_exc = []
hist_mean_inh = []
spike_times_exc = []
spike_times_inh = []
mean_exc = []
x_exc = []
mean_inh = []
x_inh = []

# Loop through the runs
for i in range(1, num_runs + 1):
    nodes_ex_i, nodes_in_i, bin_centers, hist_mean_exc_i, hist_mean_inh_i, spike_times_exc_i, spike_times_inh_i, \
        mean_exc_i, x_exc_i, mean_inh_i, x_inh_i = run_sim(i * 123, plotting_flag, i > num_runs/2)
    nodes_ex.append(nodes_ex_i)
    nodes_in.append(nodes_in_i)
    hist_mean_exc.append(hist_mean_exc_i)
    hist_mean_inh.append(hist_mean_inh_i)
    spike_times_exc.append(spike_times_exc_i)
    spike_times_inh.append(spike_times_inh_i)
    mean_exc.append(mean_exc_i)
    x_exc.append(x_exc_i)
    mean_inh.append(mean_inh_i)
    x_inh.append(x_inh_i)


# Calculate averages for hist_mean_exc and hist_mean_inh
avg_hist_exc = np.column_stack(hist_mean_exc).mean(axis=1)
avg_hist_inh = np.column_stack(hist_mean_inh).mean(axis=1)

# Calculate max values and indices
avg_hist_exc_diff = np.diff(avg_hist_exc)
avg_hist_exc_max_index = np.argmax(avg_hist_exc_diff)
avg_hist_exc_max_value = avg_hist_exc[avg_hist_exc_max_index + 1]
avg_hist_inh_diff = np.diff(avg_hist_inh)
avg_hist_inh_max_index = np.argmax(avg_hist_inh_diff)
avg_hist_inh_max_value = avg_hist_inh[avg_hist_inh_max_index + 1]

# Calculate averages and perform further analysis
smoothed_data_exc = analyzer.smoothing_kernel(avg_hist_exc)
smoothed_data_inh = analyzer.smoothing_kernel(avg_hist_inh)

# Plot the average firing rate over time
# Excitatory Neurons
m_plot.plot_avg_firing_rate(bin_centers, avg_hist_exc, smoothed_data_exc, 0)
# Inhibitory Neurons
m_plot.plot_avg_firing_rate(bin_centers, avg_hist_inh, smoothed_data_inh, 1)


ind = num_runs//2
print(ind)

# Continue with the rest of your analysis or code as needed
# Data for perturbation and no perturbation cases for both types of neurons
pp_nopert_exc = np.column_stack(mean_exc[0:ind]).mean(axis=1)

pp_pert_exc = np.column_stack(mean_exc[ind:num_runs]).mean(axis=1)

pp_nopert_inh = np.column_stack(mean_inh[0:ind]).mean(axis=1)

pp_pert_inh = np.column_stack(mean_inh[ind:num_runs]).mean(axis=1)


#Degree values for x axis
x_values_exc = x_exc[0:1]
x_values_inh = x_inh[0:1]

# Divide each element by 180.0 and apply np.pi
x_values_exc = [x * 180.0 / np.pi  for x in x_values_exc]
x_values_inh = [x * 180.0 / np.pi  for x in x_values_inh]

# Convert them to numpy array
np.array(x_values_exc)
np.array(x_values_inh)


plt.figure()
# Plot the results. average firing rate vs. orientation preference
plt.plot(np.squeeze(x_values_exc), analyzer.smoothing_kernel(pp_nopert_exc), label="nosim")
plt.plot(np.squeeze(x_values_exc), analyzer.smoothing_kernel(pp_pert_exc), label="sim")
plt.xlabel("Orientation Preference in Degrees")
plt.ylabel("Average firing rate of neurons")
plt.title("Stim Angle: {} {} {}".format(preferred_direction*180/np.pi, a, g))
plt.legend()

plt.figure()
plt.plot(np.squeeze(x_values_inh), analyzer.smoothing_kernel(pp_nopert_inh), label="nosim")
plt.plot(np.squeeze(x_values_inh), analyzer.smoothing_kernel(pp_pert_inh), label="sim")
plt.xlabel("Orientation Preference in Degrees")
plt.ylabel("Average firing rate of neurons")
plt.title("Stim Angle: {} {} {}".format(preferred_direction*180/np.pi, a, g))
plt.legend()


# To dave the datas
g = -1*g
filename = f'my_arrays_a_{a}_g_{g}.npz'

np.savez(filename, array1=np.squeeze(x_values_exc), array2=analyzer.smoothing_kernel(pp_nopert_exc), array3=analyzer.smoothing_kernel(pp_pert_exc), \
         array4=analyzer.smoothing_kernel(pp_nopert_inh), array5=analyzer.smoothing_kernel(pp_pert_inh))
plt.show()
plt.close()







