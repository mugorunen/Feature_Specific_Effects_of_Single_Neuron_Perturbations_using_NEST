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

simtime = 20000.0  # Simulation time in ms
order = 400

# Define Simulation Parameters
bin_width = 200.0
delay = 1.5  # synaptic delay in ms
g = 2.0  # ratio inhibitory weight/excitatory weight
eta = 2.0  # external rate relative to threshold rate
epsilon = 0.1  # connection probability
NE = order  # number of excitatory neurons
NI = order  # number of inhibitory neurons
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
J_in = -g * J_ex  # amplitude of inhibitory postsynaptic potential

connection_seed=70

# Define parameters
preferred_direction = 0*np.pi / 4  # Preferred direction (e.g., 45 degrees)

# Generate stimulus directions from 0 to 2*pi radians
pyrng_gen = np.random.RandomState(connection_seed)

#pref_angles_p_exc = pyrng_gen.uniform(-np.pi/2, np.pi/2, NE)
#pref_angles_p_inh = pyrng_gen.uniform(-np.pi/2, np.pi/2, NI)
pref_angles_p_exc = np.linspace(-np.pi/2, np.pi/2, NE)
pref_angles_p_inh = np.linspace(-np.pi/2, np.pi/2, NI)

#SEED DEGISTIRMEK ETKI EDER MI?
np.random.seed(13)
np.random.shuffle(pref_angles_p_exc)
np.random.shuffle(pref_angles_p_inh)
kappa_1=100
# Generate the tuning curve
tuning_curve_exc = vonmises.pdf(2*pref_angles_p_exc, kappa_1, 2*preferred_direction)
tuning_curve_inh = vonmises.pdf(2*pref_angles_p_inh, kappa_1, 2*preferred_direction)



I_max = 4000

ww_EE = np.zeros((NE, NE))
ww_EI = np.zeros((NE, NI))
ww_IE = np.zeros((NI, NE))
ww_II = np.zeros((NI, NI))

A=1
k=4



kappa_2 = 10
a=3
g=-2
p = 0.1
for i in range(NE):
    ww_EE[:, i] = p*vonmises.pdf(2*pref_angles_p_exc, kappa_2, 2*pref_angles_p_exc[i])
    

for i in range(NI):
    ww_EI[:, i] = a*p*vonmises.pdf(2*pref_angles_p_exc, kappa_2, 2*pref_angles_p_inh[i])

for i in range(NE):
    ww_IE[:, i] = g*p*vonmises.pdf(2*pref_angles_p_inh, kappa_2, 2*pref_angles_p_exc[i])

for i in range(NI):
    ww_II[:, i] = g*p*vonmises.pdf(2*pref_angles_p_inh, kappa_2, 2*pref_angles_p_inh[i])

analyzer = NetworkAnalyzer(NE, NI, N_neurons, simtime, bin_width)
m_plot = PlottingFuncs(N_neurons, simtime, bin_width, CE, CI)

print(np.max(ww_EE))

def run_sim(random_seed, plotting_flag, sim):
    # Reset previous simulations
    nest.ResetKernel()
    # Set the number of threads you want to use
    num_threads = 10
    # Set the kernel status to change the numbers of threads
    nest.SetKernelStatus({"local_num_threads": num_threads})
    # Set connection seed
    nest.SetKernelStatus({"rng_seed": connection_seed})
    dt = 0.1  # the resolution in ms
    nest.resolution = dt
    nest.print_time = True
    nest.overwrite_files = True


    rates_exc = I_max/4
    rates_inh = I_max/4


    print("Building network")
    #Define Connection Parameters
    #conn_params_ex = {"rule": "fixed_indegree", "indegree": CE}
    #conn_params_in = {"rule": "fixed_indegree", "indegree": CI}
    conn_params_ex = {"rule": "all_to_all"}
    conn_params_in = {"rule": "all_to_all"}
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
    noise_exc = nest.Create("poisson_generator", params={"rate": rates_exc})
    noise_inh = nest.Create("poisson_generator", params={"rate": rates_inh})
    #noise = nest.Create("poisson_generator", params={"rate": p_rate})

    # Connect Noise Generators 
    nest.Connect(noise_exc, nodes_ex)
    nest.Connect(noise_inh, nodes_in)
    #nest.Connect(noise, nodes_ex, syn_spec="excitatory")
    #nest.Connect(noise, nodes_in, syn_spec="excitatory")

    # Connect recorders
    nest.Connect(nodes_ex, espikes)
    nest.Connect(nodes_in, ispikes)


    #kl = abs(preferred_direction-pref_angles_p_exc)
    #min_arr = np.argmin(kl)
    #print(pref_angles_p_exc[min_arr])
#
    #src_id=min_arr+1    

    abs_array = abs(preferred_direction-pref_angles_p_exc)
    sorted_indices = np.argsort(abs_array)
    src_indices = sorted_indices[0:10]

    abs_array_inh = abs(preferred_direction-pref_angles_p_inh)
    sorted_indices_inh = np.argsort(abs_array_inh)
    src_indices_inh = sorted_indices_inh[0:1]
    #src_indices = np.array([10, 70, 150, 230, 280])
    # 20.0, 5.0 2.5
    base_amplitude=10.0
    # Create Simulator and Connect it

    src_id = src_indices+1

 
    t=0.1
    amplitude_exc= vonmises.pdf(2*pref_angles_p_exc, kappa_1, 2*pref_angles_p_exc[src_indices[0]])
    amplitude_exc = t*amplitude_exc*base_amplitude/np.max(amplitude_exc)
  
    amplitude_inh= vonmises.pdf(2*pref_angles_p_inh, kappa_1, 2*pref_angles_p_inh[src_indices_inh[0]])
    amplitude_inh = t*amplitude_inh*base_amplitude/np.max(amplitude_inh)


    #plt.figure()
    #plt.plot(pref_angles_p_exc, amplitude_exc)
    #plt.xlabel("Stimulus Direction (radians)")
    #plt.ylabel("Current")
    #plt.title("von Mises Tuning Curve")
    #
    #
    #
    #
    #
    #plt.show()


    
    
    stim_params_1 = {"amplitude": 160.0, "start": 4000.0, "stop": simtime}
    stim_1= nest.Create("dc_generator", params=stim_params_1)


    # Connect the stimulator to the neuron
    #for i in range(NE):
    #        stim_params_exc = {"amplitude": amplitude_exc[i], "start": 5150.0, "stop": 48150.0}
    #        stimulator_exc = nest.Create("dc_generator", params=stim_params_exc)
    #        nest.Connect(stimulator_exc, nodes_ex[i])
    #for i in range(NI):        
    #    stim_params_inh = {"amplitude": amplitude_inh[i], "start": 5150.0, "stop": 48150.0}
    #    stimulator_inh = nest.Create("dc_generator", params=stim_params_inh)
    #    nest.Connect(stimulator_inh, nodes_in[i])
    
    if sim==True:
            
        
        

        for k in range(len(src_id)):
            print(src_id)
            nest.Connect(stim_1, nodes_ex[src_indices[k]])
        
    #ctr, src_id, targets_exc, targets_inh = analyzer.find_src_target_ids(nodes_ex, nodes_in)
    

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

    #m_plot.plot_raster_plot(sr1_spikes, sr1_times, sr2_spikes, sr2_times)
    #plt.plot()
    #plt.show()

    
    


    # Calculate avg. firing rates of excitatory neurons
    avg_firing_exc, spike_times_exc = analyzer.calculate_avg_firing(nodes_ex, simtime, sr1_spikes, sr1_times, 0)
    print(avg_firing_exc)
    # Calculate avg. firing rates of inhibitory neurons
    avg_firing_inh, spike_times_inh = analyzer.calculate_avg_firing(nodes_in, simtime, sr2_spikes, sr2_times, 1)
    print(avg_firing_inh)



    # Calculate CoV of excitatory neurons
    CoV_exc = analyzer.calculate_CoV(spike_times_exc)
    # Calculate CoV of inhibitory neurons
    CoV_inh = analyzer.calculate_CoV(spike_times_inh)

    # Calculating firing rates for both populations
    hist_counts_all_exc, bin_centers_exc, avg_hist_counts_exc = analyzer.calculating_firing_rates(src_id, spike_times_exc, 0)

    hist_counts_all_inh, bin_centers_inh, avg_hist_counts_inh = analyzer.calculating_firing_rates(src_id, spike_times_inh, 1)

    smoothed_data_exc = analyzer.smoothing_kernel(avg_hist_counts_exc)
    smoothed_data_inh = analyzer.smoothing_kernel(avg_hist_counts_inh)


    ss = np.mean(hist_counts_all_exc, axis=1)
    ss = np.squeeze(ss)
    #ll = np.ones(400)*stim_angle[src_id-1]
    diff = pref_angles_p_exc
    diff_new = np.delete(diff, src_indices)

    diff_new_indices = np.argsort(diff_new)
    #plt.figure()
    #plt.plot(diff_new[diff_new_indices], ss[diff_new_indices])


    if (plotting_flag):
        # Plot of connection of source and target neuron (in our case central neuron ctr)

        #m_plot.plot_spatial_connections(nodes_ex, ctr)
        ## Plot source connectivity matrix
        #m_plot.plot_connectivity(connectivity_target_matrix)
        #m_plot.plot_connectivity(connectivity_01)
        #m_plot.plot_connectivity(connectivity_second_degree)
        #m_plot.plot_connectivity(connectivity_third_degree)
        # Plot raster plot
        #m_plot.plot_raster_plot(sr1_spikes, sr1_times, sr2_spikes, sr2_times)
        # Plot CV of excitatory neurons
        #m_plot.plot_CV_plot(CoV_exc, 0)
        # Plot CV of inhibitory neurons
        #m_plot.plot_CV_plot(CoV_inh, 1)
        # Plot histogram plot of perturbed neuron
        #m_plot.plot_hist_perturbed(spike_times_exc, src_id)
        # Plot average firing rate of excitatory neurons connected to the perturbed neuron
        m_plot.plot_avg_firing_rate(bin_centers_exc, avg_hist_counts_exc, smoothed_data_exc, 0)
        # Plot average firing rate of inhibitory neurons connected to the perturbed neuron
        #m_plot.plot_avg_firing_rate(bin_centers_inh, avg_hist_counts_inh, smoothed_data_inh, 1)
        # Plot one example of excitatory neuron connected to the perturbed neuron
        #m_plot.plot_example_neuron(bin_centers_exc, deneme[0].T, analyzer.smoothing_kernel(deneme[0].T), 0)
        # Plot one example of inhibitory neuron connected to the perturbed neuron
        #m_plot.plot_example_neuron(bin_centers_inh, deneme[1].T, analyzer.smoothing_kernel(deneme[1]).T, 1)


        #plt.show()

    return nodes_ex, nodes_in, bin_centers_exc, avg_hist_counts_exc, avg_hist_counts_inh, spike_times_exc, spike_times_inh, ss[diff_new_indices], diff_new[diff_new_indices]







plotting_flag = False
# Define the number of runs
num_runs = 6

# Lists to store the results
nodes_ex = []
nodes_in = []
hist_mean_exc = []
hist_mean_inh = []
spike_times_exc = []
spike_times_inh = []
ss = []
dd = []

# Loop through the runs
for i in range(1, num_runs + 1):
    nodes_ex_i, nodes_in_i, bin_centers, hist_mean_exc_i, hist_mean_inh_i, spike_times_exc_i, spike_times_inh_i, ss_i, dd_i = run_sim(i * 123, plotting_flag, i > num_runs/2)
    nodes_ex.append(nodes_ex_i)
    nodes_in.append(nodes_in_i)
    hist_mean_exc.append(hist_mean_exc_i)
    hist_mean_inh.append(hist_mean_inh_i)
    spike_times_exc.append(spike_times_exc_i)
    spike_times_inh.append(spike_times_inh_i)
    ss.append(ss_i)
    dd.append(dd_i)


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

print("Maximum firing rate at the onset (Exc):", avg_hist_exc_max_value)
print("Time value at max value (Exc):", bin_centers[avg_hist_exc_max_index + 1])
print("Maximum firing rate at the onset (Inh):", avg_hist_inh_max_value)
print("Time value at max value (Inh):", bin_centers[avg_hist_inh_max_index + 1])

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
pp_nosim = np.column_stack(ss[0:ind]).mean(axis=1)

pp_sim = np.column_stack(ss[ind:num_runs]).mean(axis=1)


# Assuming dd[1:2] is a list of values
dd_values = dd[0:1]
#plt.figure()
#plt.plot(np.squeeze(dd_values), analyzer.smoothing_kernel(np.column_stack(ss[0:1]).mean(axis=1)), label="1")
#plt.plot(np.squeeze(dd_values), analyzer.smoothing_kernel(np.column_stack(ss[2:3]).mean(axis=1)), label="2")
#plt.plot(np.squeeze(dd_values), analyzer.smoothing_kernel(np.column_stack(ss[3:4]).mean(axis=1)), label="3")
#plt.plot(np.squeeze(dd_values), analyzer.smoothing_kernel(np.column_stack(ss[4:5]).mean(axis=1)), label="4")

# Divide each element by 180.0 and apply np.pi
dd_values = [x * 180.0 / np.pi  for x in dd_values]
np.array(dd_values)
plt.figure()
#plt.plot(dd_nosim*180/np.pi, pp)
plt.plot(np.squeeze(dd_values), analyzer.smoothing_kernel(pp_nosim), label="nosim")
plt.plot(np.squeeze(dd_values), analyzer.smoothing_kernel(pp_sim), label="sim")
plt.xlabel("Orientation Preference in Degrees")
plt.ylabel("Average firing rate of neurons")
plt.title("Stim Angle: {} {} {}".format(preferred_direction*180/np.pi, a, g))
plt.legend()

#plt.figure()
#plt.plot(np.squeeze(dd_values), analyzer.smoothing_kernel(pp_sim)-analyzer.smoothing_kernel(pp_nosim))
#plt.xlabel("Orientation Preference in Degrees")
#plt.ylabel("Firing rate difference")
#plt.title("Stim Angle: {} {} {}".format(preferred_direction*180/np.pi, a, g))

#print(analyzer.smoothing_kernel(pp_sim)-analyzer.smoothing_kernel(pp_nosim))
plt.show()
plt.close()







