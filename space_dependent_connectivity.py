# Import necessary libraries
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, LogNorm
from matplotlib.cm import ScalarMappable
import elephant.statistics as es
from quantities import ms
import neo
import nest
import nest.raster_plot


# Reset previous simulations
nest.ResetKernel()
# Set the number of threads you want to use
num_threads = 4
# Set the kernel status to change the number of threads
nest.SetKernelStatus({"local_num_threads": num_threads})
connection_seed = 42
nest.SetKernelStatus({"rng_seed": connection_seed})
dt = 0.1  # the resolution in ms
nest.resolution = dt
nest.print_time = True
nest.overwrite_files = True



# Define Simulation Parameters
startbuild = time.time()
simtime = 20000.0  # Simulation time in ms
delay = 1.5  # synaptic delay in ms
g = 5.0  # ratio inhibitory weight/excitatory weight
eta = 2.0  # external rate relative to threshold rate
epsilon = 0.1  # connection probability
order = 100
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





print("Building network")
#Define Connection Parameters
conn_params_ex = {"rule": "fixed_indegree", "indegree": CE, "mask": {"circular": {"radius": 1.5}}}
conn_params_in = {"rule": "fixed_indegree", "indegree": CI, "mask": {"circular": {"radius": 1.5}}}
#Define the Positions
pos_ex = nest.spatial.free(pos=nest.random.uniform(min=-2.0, max=2.0), num_dimensions=2)
# Create excitatory neurons, inhibitory neurons, poisson spike generator, and spike recorders
nodes_ex = nest.Create("iaf_psc_delta", NE, params=neuron_params, positions=pos_ex)
nodes_in = nest.Create("iaf_psc_delta", NI, params=neuron_params, positions=pos_ex)
noise = nest.Create("poisson_generator", params={"rate": p_rate})
espikes = nest.Create("spike_recorder")
ispikes = nest.Create("spike_recorder")

#Define the Synapses
nest.CopyModel("static_synapse", "excitatory", {"weight": J_ex, "delay": delay})
nest.CopyModel("static_synapse", "inhibitory", {"weight": J_in, "delay": delay})

# Create Connections between populations
nest.Connect(nodes_ex, nodes_ex, conn_params_ex, "excitatory")
nest.Connect(nodes_ex, nodes_in, conn_params_ex, "excitatory")

nest.Connect(nodes_in, nodes_ex, conn_params_in, "inhibitory")
nest.Connect(nodes_in, nodes_in, conn_params_in, "inhibitory")

# Connect Noise Generators 
nest.Connect(noise, nodes_ex, syn_spec="excitatory")
nest.Connect(noise, nodes_in, syn_spec="excitatory")

# Connect recorders
nest.Connect(nodes_ex, espikes, syn_spec="excitatory")
nest.Connect(nodes_in, ispikes, syn_spec="excitatory")






# Plot Target Neurons of the Center Neuron
fig1 = nest.PlotLayer(nodes_ex, nodesize=50)

ctr = nest.FindCenterElement(nodes_ex)
nest.PlotTargets(
    ctr,
    nodes_ex,
    fig=fig1,
    src_size=250,
    tgt_color="red",
    tgt_size=20,
    mask_color="red",
    probability_cmap="Greens",
)

plt.xticks(np.arange(-1.5, 1.55, 0.5))
plt.yticks(np.arange(-1.5, 1.55, 0.5))
plt.grid(True)
plt.axis([-2.0, 2.0, -2.0, 2.0])
plt.title("Connection targets")

# Plot Source Neurons of the Center Neuron
fig2 = nest.PlotLayer(nodes_ex)
nest.PlotSources(
    nodes_ex,
    ctr,
    fig=fig2,
    src_size=50,
    src_color='green',
    tgt_size=20,
    mask_color="red",
    probability_cmap="Greens",
)

plt.xticks(np.arange(-1.5, 1.55, 0.5))
plt.yticks(np.arange(-1.5, 1.55, 0.5))
plt.grid(True)
plt.axis([-2.0, 2.0, -2.0, 2.0])
plt.title("Connection targets")





# target ids start from 1
# Inhibitory and Excitatory Target Neurons
target_ids_exc = nest.GetTargetNodes(ctr, nodes_ex)[0]
target_ids_inh = nest.GetTargetNodes(ctr, nodes_in)[0]
targets_exc = target_ids_exc.tolist()
targets_inh = target_ids_inh.tolist()
print(targets_exc)
print(targets_inh)

# ID of the center neuron
nest.GetLocalNodeCollection(ctr)
src_id = int(ctr.tolist()[0])
print("src_id:", src_id)




# Create Simulator and Connect it
amplitude=160.0
stim_params = {"amplitude": amplitude, "start": 5150.0, "stop": 18150.0}
stimulator = nest.Create("dc_generator", params=stim_params)

#Choose Some Neurons
exc_id1 = int(targets_exc[0])
inh_id1 = int(targets_inh[0])
inh_id2 = int(targets_inh[1])

print("inh_id1=", inh_id1)
print("inh_id2=", inh_id2)

# Connect the stimulator to the neuron
nest.Connect(stimulator, nodes_ex[src_id-1])
#nest.Connect(stimulator, nodes_ex[exc_id1-1])
#nest.Connect(stimulator, nodes_in[inh_id1-NE-1])
#nest.Connect(stimulator, nodes_in[inh_id2-NE-1])
endbuild = time.time()

connectivity=np.zeros((N_neurons,N_neurons))

conn_ex=nest.GetConnections(nodes_ex)
conn_ex_source= nest.GetStatus(conn_ex, keys='source')
conn_ex_target= nest.GetStatus(conn_ex, keys='target')
conn_ex_weight= nest.GetStatus(conn_ex, keys='weight')

conn_in=nest.GetConnections(nodes_in)
conn_in_source= nest.GetStatus(conn_in, keys='source')
conn_in_target= nest.GetStatus(conn_in, keys='target')
conn_in_weight= nest.GetStatus(conn_in, keys='weight')

for i in range(len(conn_ex_source)):
	if conn_ex_source[i]<= N_neurons and conn_ex_target[i]<= N_neurons:
		connectivity[conn_ex_source[i]-1,conn_ex_target[i]-1]=conn_ex_weight[i]
for i in range(len(conn_in_source)):
	if conn_in_source[i]<=N_neurons and conn_in_target[i]<= N_neurons:
		connectivity[conn_in_source[i]-1,conn_in_target[i]-1]=conn_in_weight[i]
		
connectivity_matrix=connectivity.T
np.savetxt("connectivity.dat",connectivity_matrix,delimiter="\t",fmt="%1.4f")
# Normalize the values for colormap mapping
norm = Normalize(vmin=np.min(connectivity_matrix), vmax=np.max(connectivity_matrix))
colormap = plt.get_cmap('viridis')  # Choose a colormap (e.g., 'viridis')

# Create the connection plot
plt.figure(figsize=(10, 10))
for row in range(connectivity_matrix.shape[0]):
    for col in range(connectivity_matrix.shape[1]):
        if connectivity_matrix[row, col] != 0:  # Plot only connections with non-zero values
            color = colormap(norm(connectivity_matrix[row, col]))
            plt.plot([col], [500 - row], marker='o', markersize=5, color=color)

# Customize the plot
sm = ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])  # Prevents error in colorbar
plt.colorbar(sm, label='Amplitude')  # Add colorbar with amplitude label
plt.xlim(0, 500)
plt.ylim(0, 500)
plt.gca().invert_yaxis()
plt.title("Connection Plot with Colormap")
plt.xlabel("Column Index")
plt.ylabel("Row Index")
plt.show()



# Start Simulation
print("Simulating")

nest.Simulate(simtime)

endsimulate = time.time()


# Extract Some Parameters from the Simulation
events_ex = espikes.n_events
events_in = ispikes.n_events

rate_ex = events_ex / simtime * 1000.0 / NE
rate_in = events_in / simtime * 1000.0 / NI

num_synapses_ex = nest.GetDefaults("excitatory")["num_connections"]
num_synapses_in = nest.GetDefaults("inhibitory")["num_connections"]
num_synapses = num_synapses_ex + num_synapses_in

build_time = endbuild - startbuild
sim_time = endsimulate - endbuild




# Extract spikes and plot raster plot
sr1_spikes = espikes.events['senders']
sr1_times = espikes.events['times']
sr2_spikes = ispikes.events['senders']
sr2_times = ispikes.events['times']

plt.figure('Figure 3')

plt.plot(sr1_times, sr1_spikes,
        '.', markersize=1, color='blue', label='Exc')
plt.plot(sr2_times, sr2_spikes,
        '.', markersize=1, color='orange', label='Inh')
plt.xlabel('time (ms)')
plt.ylabel('neuron id')
plt.legend()




# Calculation of the average firing rate
def calculate_avg_firing(pop, sim_time, spikes, times, flag):
    #min_value = np.min(spikes)
    #max_value = np.max(spikes)
    if flag==0:
        neuron_ids = [i for i in range(1, NE + 1, 1)]
    else:
        neuron_ids = [i for i in range(1 + NE, NI + 1 + NE, 1)]
    firing_rates = []
    spike_times = []
    for neuron_id in neuron_ids:
        spikes_for_neuron = np.sum(spikes == neuron_id)
        firing_rate = spikes_for_neuron / (sim_time / 1000)  # Convert to Hz
        firing_rates.append(firing_rate)

        time_indices = np.where(spikes==neuron_id)[0]
        time_values = times[time_indices]
        spike_times.append(time_values)

        
    # Calculate average firing rate
    average_firing_rate = np.mean(firing_rates)
    
    return average_firing_rate, spike_times


# Calculate avg. firing rates of excitatory neurons
avg_firing_exc, spike_times_exc = calculate_avg_firing(nodes_ex, simtime, sr1_spikes, sr1_times, 0)
print(avg_firing_exc)
# Calculate avg. firing rates of inhibitory neurons
avg_firing_inh, spike_times_inh = calculate_avg_firing(nodes_in, simtime, sr2_spikes, sr2_times, 1)
print(avg_firing_inh)





# Calculate Coefficient of Variation values of Neurons and plot histogram of them
def calculate_CoV(spike_times):
    consecutive_diffs = [np.diff(sublist) for sublist in spike_times]
    mean_isi = [np.mean(sublist) for sublist in consecutive_diffs]
    std_isi = [np.std(sublist) for sublist in consecutive_diffs]

    CoV = [a/b for a, b in zip(std_isi, mean_isi)]

    return CoV

# Calculate CoV of excitatory neurons
CoV_exc = calculate_CoV(spike_times_exc)
# Calculate CoV of inhibitory neurons
CoV_inh = calculate_CoV(spike_times_inh)

# Create a histogram plot for excitatory neurons
plt.figure('Figure 4')
plt.hist(CoV_exc, bins=30, edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("CV of Excitatory Neurons")

# Create a histogram plot for inhibitory neurons
plt.figure('Figure 5')
plt.hist(CoV_inh, bins=30, edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("CV of Inhibitory Neurons")






spike_times_exc = np.array(spike_times_exc)
print(spike_times_exc.shape)
bin_width = 200.0
#Plot the histogram of the Perturbed Neuron
plt.figure('Figure 6')
# Create a histogram plot for excitatory neurons
plt.hist(spike_times_exc[src_id-1], bins=int(simtime/bin_width), edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of the Stimulated Neuron")






# Plot the mean firing rate of the Excitatory Neurons Connected to the Perturbed One
hist_counts_all_exc = []
elep_spike_times_exc=[]
#targets_exc.remove(src_id)
for i in targets_exc:
    if i==src_id:
        data = spike_times_exc[i-1]
        num_bins = int(simtime/bin_width)
        hist_counts_kk, hist_edges = np.histogram(data, bins=num_bins, range=(min(data), max(data)))
    else:
        data = spike_times_exc[i-1]
        elep_spike_times_exc.append(data)
        num_bins = int(simtime/bin_width)
        hist_counts, hist_edges = np.histogram(data, bins=num_bins, range=(min(data), max(data)))
        hist_counts_all_exc.append(hist_counts/0.2)

bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
hist_counts_all_exc = np.array(hist_counts_all_exc)
print(hist_counts_all_exc.shape)

plt.figure('Figure 7')
plt.plot(bin_centers, hist_counts_all_exc.T.mean(axis=1))
#plt.legend(targets_exc)
plt.title("Stim: Excitatory Neurons (CE= {}, CI= {})".format(CE, CI))
plt.xlabel('Time')
plt.ylabel('Firing Rate (Hz)')







# Plot the mean firing rate of the Excitatory Neurons Connected to the Perturbed One Using Smoothing Kernel
elep_spike_times_exc = np.array(elep_spike_times_exc)*ms
spiketrains = [neo.SpikeTrain(times, t_start=0 * ms, t_stop=simtime * ms) for times in elep_spike_times_exc]

# Specify the parameters for the instantaneous_rate function
sampling_period = 1 * ms
kernel = 'auto'  # You can specify a specific kernel shape if needed
cutoff = 5.0 

# Call the instantaneous_rate function for each neuron's spike train
firing_rates_exc = []
for spiketrain in spiketrains:
    rate_analog_signal = es.instantaneous_rate(spiketrain, sampling_period, kernel=kernel, cutoff=cutoff)
    firing_rates_exc.append(rate_analog_signal.magnitude)

# Calculate the average firing rate across neurons for each time step
average_firing_rate = np.mean(firing_rates_exc, axis=0)
print(average_firing_rate.shape)

# Create a time axis based on the first spiketrain's duration
time_axis = np.linspace(0, simtime, len(average_firing_rate))

# Plot the average firing rate over time
plt.plot(time_axis, average_firing_rate, color='blue')
plt.grid(True)
plt.figure('Figure 8')
plt.plot(bin_centers, hist_counts_all_exc[2].T)
#plt.legend(targets_exc)
plt.title("Stim: Excitatory Neurons (CE= {}, CI= {})".format(CE, CI))
plt.xlabel('Time')
plt.ylabel('Firing Rate (Hz)')
# Plot the average firing rate over time
plt.plot(time_axis, firing_rates_exc[2], color='blue')
plt.grid(True)







# Plot the mean firing rate of the Inhibitory Neurons Connected to the Perturbed One
hist_counts_all_inh = []
elep_spike_times_inh = []
for i in targets_inh:
    data = spike_times_inh[i-NE-1]
    elep_spike_times_inh.append(data)
    num_bins = int(simtime/bin_width)
    hist_counts, hist_edges = np.histogram(data, bins=num_bins, range=(min(data), max(data)))
    hist_counts_all_inh.append(hist_counts/0.2)
bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
hist_counts_all_inh = np.array(hist_counts_all_inh)

plt.figure('Figure 9')
plt.plot(bin_centers, hist_counts_all_inh.T.mean(axis=1))
#plt.legend(targets_inh)
plt.title("Stim: Inhibitory Neurons (CE= {}, CI= {})".format(CE, CI))
plt.xlabel('Time')
plt.ylabel('Firing Rate (Hz)')





# Plot the mean firing rate of the Inhibitory Neurons Connected to the Perturbed One using smoothing Kernel
elep_spike_times_inh = np.array(elep_spike_times_inh)*ms

spiketrains = [neo.SpikeTrain(times, t_start=0 * ms, t_stop=simtime * ms) for times in elep_spike_times_inh]

# Specify the parameters for the instantaneous_rate function
sampling_period = 1 * ms
kernel = 'auto'  # You can specify a specific kernel shape if needed
cutoff = 5.0 

# Call the instantaneous_rate function for each neuron's spike train
firing_rates_inh = []
for spiketrain in spiketrains:
    rate_analog_signal = es.instantaneous_rate(spiketrain, sampling_period, kernel=kernel, cutoff=cutoff)
    firing_rates_inh.append(rate_analog_signal.magnitude)

# Calculate the average firing rate across neurons for each time step
average_firing_rate = np.mean(firing_rates_inh, axis=0)
print(average_firing_rate.shape)
# Create a time axis based on the first spiketrain's duration
time_axis = np.linspace(0, simtime, len(average_firing_rate))

# Plot the average firing rate over time
plt.plot(time_axis, average_firing_rate, color='blue')
plt.grid(True)

plt.figure('Figure 10')
plt.plot(bin_centers, hist_counts_all_inh[2].T)
#plt.legend(targets_exc)
plt.title("Stim: Inhibitory Neurons (CE= {}, CI= {})".format(CE, CI))
plt.xlabel('Time')
plt.ylabel('Firing Rate (Hz)')

# Plot the average firing rate over time
plt.plot(time_axis, firing_rates_inh[2], color='blue')
plt.grid(True)







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
'''



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



plt.show()




