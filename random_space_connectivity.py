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
import networkx as nx
import nest
import nest.raster_plot


simtime = 10000.0  # Simulation time in ms
order = 500

# Define Simulation Parameters
bin_width = 200.0
delay = 1.5  # synaptic delay in ms
g = 5.0  # ratio inhibitory weight/excitatory weight
eta = 2.0  # external rate relative to threshold rate
epsilon = 0.1  # connection probability
NE = 4 * order  # number of excitatory neurons
NI = 1 * order  # number of inhibitory neurons
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
J = 0.5  # postsynaptic amplitude in mV
J_ex = J  # amplitude of excitatory postsynaptic potential
J_in = -g * J_ex  # amplitude of inhibitory postsynaptic potential
nu_th = theta / (J * CE * tauMem)
nu_ex = eta * nu_th
p_rate = 1000.0 * nu_ex * CE




def run_sim(random_seed, plotting_flag):
    # Reset previous simulations
    nest.ResetKernel()
    # Set the number of threads you want to use
    num_threads = 4
    # Set the kernel status to change the number of threads
    nest.SetKernelStatus({"local_num_threads": num_threads})
    dt = 0.1  # the resolution in ms
    nest.resolution = dt
    nest.print_time = True
    nest.overwrite_files = True
    

    print("Building network")
    #Define Connection Parameters
    conn_params_ex = {"rule": "fixed_indegree", "indegree": CE}
    conn_params_in = {"rule": "fixed_indegree", "indegree": CI}
    #Define the Positions
    pos_ex = nest.spatial.free(pos=nest.random.uniform(min=-2.0, max=2.0), num_dimensions=2)
    # Create excitatory neurons, inhibitory neurons, poisson spike generator, and spike recorders
    nodes_ex = nest.Create("iaf_psc_delta", NE, params=neuron_params, positions=pos_ex)
    nodes_in = nest.Create("iaf_psc_delta", NI, params=neuron_params, positions=pos_ex)
    nest.SetKernelStatus({"rng_seed": random_seed})
    noise = nest.Create("poisson_generator", params={"rate": p_rate})
    espikes = nest.Create("spike_recorder")
    ispikes = nest.Create("spike_recorder")

    #Define the Synapses
    nest.CopyModel("static_synapse", "excitatory", {"weight": J_ex, "delay": delay})
    nest.CopyModel("static_synapse", "inhibitory", {"weight": J_in, "delay": delay})

    # Create Connections between populations
    nest.Connect(nodes_ex, nodes_ex, conn_params_ex, syn_spec="excitatory")
    nest.Connect(nodes_ex, nodes_in, conn_params_ex, syn_spec="excitatory")

    nest.Connect(nodes_in, nodes_ex, conn_params_in, "inhibitory")
    nest.Connect(nodes_in, nodes_in, conn_params_in, "inhibitory")

    # Connect Noise Generators 
    nest.Connect(noise, nodes_ex, syn_spec="excitatory")
    nest.Connect(noise, nodes_in, syn_spec="excitatory")

    # Connect recorders
    nest.Connect(nodes_ex, espikes, syn_spec="excitatory")
    nest.Connect(nodes_in, ispikes, syn_spec="excitatory")




    # Plot Target Neurons of the Center Neuron
    ctr = nest.FindCenterElement(nodes_ex)
        
    

    # target ids start from 1
    # Inhibitory and Excitatory Target Neurons

    # ID of the center neuron
    src_id = int(ctr.tolist()[0])
    print("src_id:", ctr.tolist())



    # Create Simulator and Connect it
    amplitude=160.0
    stim_params = {"amplitude": amplitude, "start": 5150.0, "stop": 18150.0}
    stimulator = nest.Create("dc_generator", params=stim_params)


    # Connect the stimulator to the neuron
    nest.Connect(stimulator, nodes_ex[src_id-1])

    target_ids_exc = nest.GetTargetNodes(ctr, nodes_ex)[0]
    target_ids_inh = nest.GetTargetNodes(ctr, nodes_in)[0]
    targets_exc = target_ids_exc.tolist()
    targets_inh = target_ids_inh.tolist()
    #print(targets_exc)
    #print(targets_inh)


    
    
    
    #connectivity_matrix = analyzer.create_connectivity(nodes_ex, nodes_in)
    #np.savetxt("connectivity.dat",connectivity_matrix,delimiter="\t",fmt="%1.4f")

    
    


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
    print(avg_firing_exc)
    # Calculate avg. firing rates of inhibitory neurons
    avg_firing_inh, spike_times_inh = analyzer.calculate_avg_firing(nodes_in, simtime, sr2_spikes, sr2_times, 1)
    print(avg_firing_inh)



    # Calculate CoV of excitatory neurons
    CoV_exc = analyzer.calculate_CoV(spike_times_exc)
    # Calculate CoV of inhibitory neurons
    CoV_inh = analyzer.calculate_CoV(spike_times_inh)


    # Calculating firing rates for both populations
    hist_counts_all_exc, bin_centers_exc, avg_hist_counts_exc, average_firing_rate_exc_elep, firing_rates_smoothed_exc_elep = analyzer.calculating_firing_rates(targets_exc, src_id, spike_times_exc, 0)

    hist_counts_all_inh, bin_centers_inh, avg_hist_counts_inh, average_firing_rate_inh_elep, firing_rates_smoothed_inh_elep = analyzer.calculating_firing_rates(targets_inh, src_id, spike_times_inh, 1)



    if (plotting_flag==True):
        # Plot of connection of source and target neuron (in our case central neuron ctr)
        m_plot.plot_spatial_connections(nodes_ex, ctr)
        # Plot source connectivity matrix
        #m_plot.plot_connectivity(connectivity_matrix)
        # Plot raster plot
        m_plot.plot_raster_plot(sr1_spikes, sr1_times, sr2_spikes, sr2_times)
        # Plot CV of excitatory neurons
        m_plot.plot_CV_plot(CoV_exc, 0)
        # Plot CV of inhibitory neurons
        m_plot.plot_CV_plot(CoV_inh, 1)
        # Plot histogram plot of perturbed neuron
        m_plot.plot_hist_perturbed(spike_times_exc, src_id)
        # Plot average firing rate of excitatory neurons connected to the perturbed neuron
        m_plot.plot_avg_firing_rate(bin_centers_exc, avg_hist_counts_exc, average_firing_rate_exc_elep, 0)
        # Plot average firing rate of inhibitory neurons connected to the perturbed neuron
        m_plot.plot_avg_firing_rate(bin_centers_inh, avg_hist_counts_inh, average_firing_rate_inh_elep, 1)
        # Plot one example of excitatory neuron connected to the perturbed neuron
        m_plot.plot_example_neuron(bin_centers_exc, hist_counts_all_exc[2].T, firing_rates_smoothed_exc_elep[2], 0)
        # Plot one example of inhibitory neuron connected to the perturbed neuron
        m_plot.plot_example_neuron(bin_centers_inh, hist_counts_all_inh[0].T, firing_rates_smoothed_inh_elep[0], 1)




    return bin_centers_exc, avg_hist_counts_exc, avg_hist_counts_inh



analyzer = NetworkAnalyzer(NE, NI, N_neurons, simtime, bin_width)
m_plot = PlottingFuncs(N_neurons, simtime, bin_width, CE, CI)

plotting_flag = False
bin_centers, hist_mean_exc_1, hist_mean_inh_1 = run_sim(1*123, plotting_flag)
bin_centers, hist_mean_exc_2, hist_mean_inh_2 = run_sim(2*123, plotting_flag)
bin_centers, hist_mean_exc_3, hist_mean_inh_3 = run_sim(3*123, plotting_flag)

avg_hist_exc = np.column_stack((hist_mean_exc_1, hist_mean_exc_2, hist_mean_exc_3)).mean(axis=1)
avg_hist_inh = np.column_stack((hist_mean_inh_1, hist_mean_inh_2, hist_mean_inh_3)).mean(axis=1)





smoothed_data_exc = analyzer.smoothing_kernel(avg_hist_exc)
smoothed_data_inh = analyzer.smoothing_kernel(avg_hist_inh)
print(smoothed_data_exc.shape)
print(smoothed_data_inh.shape)
print('Finito')
## Plot the average firing rate over time
plt.figure()
plt.plot(np.linspace(0, simtime, len(bin_centers[1:len(smoothed_data_exc)-3])), smoothed_data_exc[1:len(smoothed_data_exc)-3], color='blue')
plt.plot(bin_centers[1:len(smoothed_data_exc)-3], avg_hist_exc[:len(avg_hist_exc)-1])
plt.title('Average of Excitatory Neurons Across Trials')
plt.grid(True)


plt.figure()
plt.plot(np.linspace(0, simtime, len(bin_centers[1:len(smoothed_data_inh)-3])), smoothed_data_inh[1:len(smoothed_data_inh)-3], color='blue')
plt.plot(bin_centers[1:len(smoothed_data_inh)-3], avg_hist_exc[:len(avg_hist_inh)-1])
plt.title('Average of Inhibitory Neurons Across Trials')
plt.grid(True)

plt.show()
plt.close()


'''
# Plotting Second Edge Neurons Connected to the excitatory neuron which is target to the perturbed one 
# target ids start from 1
second_degree_target_exc = nest.GetTargetNodes(nodes_ex[targets_exc[0]-1], nodes_ex)[0]
second_degree_target_inh = nest.GetTargetNodes(nodes_ex[targets_exc[0]-1], nodes_in)[0]

second_degree_target_exc = second_degree_target_exc.tolist()
second_degree_target_inh = second_degree_target_inh.tolist()
print('second_degree_target_exc:', second_degree_target_exc)
print('second_degree_target_inh:', second_degree_target_inh)



# Plotting Firing Rate of Excitatory Neuron
data_exc = spike_times_exc[324-1]
num_bins = int(simtime/bin_width)
hist_counts, hist_edges = np.histogram(data_exc, bins=num_bins, range=(min(data), max(data)))
bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2

plt.figure('Figure 11')
plt.plot(bin_centers, hist_counts.T/0.2)
#plt.legend(targets_inh)
plt.title("Stim: Second distant Excitatory neuron (CE= {}, CI= {})".format(CE, CI))
plt.xlabel('Time')
plt.ylabel('Firing Rate (Hz)')

# Plotting Firing Rate of Excitatory Neuron Using Smoothing Kernel
data_exc = np.array(data_exc)*ms

spiketrain_sec_distant_exc = neo.SpikeTrain(data_exc, t_start=0 * ms, t_stop=simtime * ms)

# Specify the parameters for the instantaneous_rate function
sampling_period = 1 * ms
kernel = 'auto'  # You can specify a specific kernel shape if needed
cutoff = 5.0 
# Call the instantaneous_rate function for each neuron's spike train
rate_analog_signal = es.instantaneous_rate(spiketrain_sec_distant_exc, sampling_period, kernel=kernel, cutoff=cutoff)

# Extract the firing rate values and time axis
firing_rate_sec_exc = rate_analog_signal.magnitude
# Create a time axis based on the first spiketrain's duration
time_axis = np.linspace(0, simtime, len(firing_rate_sec_exc))

# Plot the average firing rate over time
plt.plot(time_axis, firing_rate_sec_exc, color='blue')
plt.grid(True)





# Plotting Firing Rate of Inhibitory Neuron
data_inh = spike_times_inh[8461-NE-1]
num_bins = int(simtime/bin_width)
hist_counts, hist_edges = np.histogram(data_inh, bins=num_bins, range=(min(data), max(data)))
bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2

plt.figure('Figure 12')
plt.plot(bin_centers, hist_counts.T/0.2)
#plt.legend(targets_inh)
plt.title("Stim: Second distant Inhibitory neuron (CE= {}, CI= {})".format(CE, CI))
plt.xlabel('Time')
plt.ylabel('Firing Rate (Hz)')


# Plotting Firing Rate of Inhibitory Neuron Using Smoothing Kernel
data_inh = np.array(data_inh)*ms

spiketrain_sec_distant_inh = neo.SpikeTrain(data_inh, t_start=0 * ms, t_stop=simtime * ms)

# Specify the parameters for the instantaneous_rate function
sampling_period = 1 * ms
kernel = 'auto'  # You can specify a specific kernel shape if needed
cutoff = 5.0 

# Call the instantaneous_rate function for each neuron's spike train
rate_analog_signal = es.instantaneous_rate(spiketrain_sec_distant_inh, sampling_period, kernel=kernel, cutoff=cutoff)

# Extract the firing rate values and time axis
firing_rate_sec_inh = rate_analog_signal.magnitude
# Create a time axis based on the first spiketrain's duration
time_axis = np.linspace(0, simtime, len(firing_rate_sec_inh))

# Plot the average firing rate over time
plt.plot(time_axis, firing_rate_sec_inh, color='blue')
plt.grid(True)












# Plotting Second Edge Neurons Connected to the inhibitory neuron which is target to the perturbed one 
# target ids start from 1
second_degree_target_exc_inh = nest.GetTargetNodes(nodes_in[targets_inh[0]-NE-1], nodes_ex)[0]
second_degree_target_inh_inh = nest.GetTargetNodes(nodes_in[targets_inh[0]-NE-1], nodes_in)[0]
second_degree_target_exc_inh = second_degree_target_exc_inh.tolist()
second_degree_target_inh_inh = second_degree_target_inh_inh.tolist()
print('second_degree_target_exc:', second_degree_target_exc_inh)
print('second_degree_target_inh:', second_degree_target_inh_inh)

# Plotting Firing Rate of Excitatory Neuron
data_exc_inh = spike_times_exc[246-1]
num_bins = int(simtime/bin_width)
hist_counts, hist_edges = np.histogram(data_exc_inh, bins=num_bins, range=(min(data), max(data)))
bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2


plt.figure('Figure 13')
plt.plot(bin_centers, hist_counts.T/0.2)
#plt.legend(targets_inh)
plt.title("Stim: Second distant Excitatory neuron (CE= {}, CI= {})".format(CE, CI))
plt.xlabel('Time')
plt.ylabel('Firing Rate (Hz)')

# Plotting Firing Rate of Excitatory Neuron Using Smoothing Kernel
data_exc_inh = np.array(data_exc_inh)*ms
spiketrain_sec_distant_exc_inh = neo.SpikeTrain(data_exc_inh, t_start=0 * ms, t_stop=simtime * ms)


# Specify the parameters for the instantaneous_rate function
sampling_period = 1 * ms
kernel = 'auto'  # You can specify a specific kernel shape if needed
cutoff = 5.0 


# Call the instantaneous_rate function for each neuron's spike train
rate_analog_signal = es.instantaneous_rate(spiketrain_sec_distant_exc_inh, sampling_period, kernel=kernel, cutoff=cutoff)
# Extract the firing rate values and time axis
firing_rate_sec_exc_inh = rate_analog_signal.magnitude
# Create a time axis based on the first spiketrain's duration
time_axis = np.linspace(0, simtime, len(firing_rate_sec_exc_inh))

# Plot the average firing rate over time
plt.plot(time_axis, firing_rate_sec_exc_inh, color='blue')
plt.grid(True)




# Plotting Firing Rate of Inhibitory Neuron
data_inh_inh = spike_times_inh[8646-NE-1]
num_bins = int(simtime/bin_width)
hist_counts, hist_edges = np.histogram(data_inh_inh, bins=num_bins, range=(min(data), max(data)))
bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2

plt.figure('Figure 14')
plt.plot(bin_centers, hist_counts.T/0.2)
#plt.legend(targets_inh)
plt.title("Stim: Second distant Inhibitory neuron (CE= {}, CI= {})".format(CE, CI))
plt.xlabel('Time')
plt.ylabel('Firing Rate (Hz)')

# Plotting Firing Rate of Inhibitory Neuron Using SMoothing Kernel
data_inh_inh = np.array(data_inh_inh)*ms
spiketrain_sec_distant_inh_inh = neo.SpikeTrain(data_inh_inh, t_start=0 * ms, t_stop=simtime * ms)

# Specify the parameters for the instantaneous_rate function
sampling_period = 1 * ms
kernel = 'auto'  # You can specify a specific kernel shape if needed
cutoff = 5.0 

# Call the instantaneous_rate function for each neuron's spike train
rate_analog_signal = es.instantaneous_rate(spiketrain_sec_distant_inh_inh, sampling_period, kernel=kernel, cutoff=cutoff)

# Extract the firing rate values and time axis
firing_rate_sec_inh_inh = rate_analog_signal.magnitude

# Create a time axis based on the first spiketrain's duration
time_axis = np.linspace(0, simtime, len(firing_rate_sec_inh_inh))
# Plot the average firing rate over time
plt.plot(time_axis, firing_rate_sec_inh_inh, color='blue')
plt.grid(True)




#Printing Network Parameters
print("Brunel network simulation (Python)")
print(f"Number of neurons : {N_neurons}")
print(f"Number of synapses: {num_synapses}")
print(f"       Excitatory : {num_synapses_ex}")
print(f"       Inhibitory : {num_synapses_in}")
print(f"Excitatory rate   : {rate_ex:.2f} Hz")
print(f"Inhibitory rate   : {rate_in:.2f} Hz")
print(f"Building time     : {build_time:.2f} s")
print(f"Simulation time   : {sim_time:.2f} s")
'''






