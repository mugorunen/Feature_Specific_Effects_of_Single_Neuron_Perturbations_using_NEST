# Import necessary libraries
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import convolve2d
from scipy.signal import convolve
from mpl_toolkits.axes_grid1 import make_axes_locatable
import elephant.statistics as es
from quantities import ms
import neo
import nest
import nest.raster_plot


def run_sim(random_seed):
    # Reset previous simulations
    nest.ResetKernel()

    # Set the number of threads you want to use
    num_threads = 4


    connection_seed = 42
    nest.SetKernelStatus({"rng_seed": connection_seed})
    nest.SetKernelStatus({"print_time": True})
    # Set the kernel status to change the number of threads
    nest.SetKernelStatus({"local_num_threads": num_threads})

    startbuild = time.time()
    dt = 0.1  # the resolution in ms
    simtime = 20000.0  # Simulation time in ms
    delay = 1.5  # synaptic delay in ms

    g = 5.0  # ratio inhibitory weight/excitatory weight
    eta = 2.0  # external rate relative to threshold rate
    epsilon = 0.1  # connection probability

    order = 100
    NE = 4 * order  # number of excitatory neurons
    NI = 1 * order  # number of inhibitory neurons
    N_neurons = NE + NI  # number of neurons in total
    N_rec = 10  # record from 50 neurons

    CE = int(epsilon * NE)  # number of excitatory synapses per neuron
    CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
    C_tot = int(CI + CE)  # total number of synapses per neuron
    print(CE, CI)

    tauMem = 20.0  # time constant of membrane potential in ms
    theta = 20.0  # membrane threshold potential in mV
    neuron_params = {"C_m": 1.0,
                     "tau_m": tauMem,
                     "t_ref": 2.0,
                     "E_L": 0.0,
                     "V_reset": 0.0,
                     "V_m": 0.0,
                     "V_th": theta}
    J = 0.4  # postsynaptic amplitude in mV
    J_ex = J  # amplitude of excitatory postsynaptic potential
    J_in = -g * J_ex  # amplitude of inhibitory postsynaptic potential


    nu_th = theta / (J * CE * tauMem)
    nu_ex = eta * nu_th
    p_rate = 1000.0 * nu_ex * CE

    print(nu_th, nu_ex, p_rate)

    nest.resolution = dt
    nest.print_time = True
    nest.overwrite_files = True

    print("Building network")

    pos_ex = nest.spatial.free(pos=nest.random.uniform(min=-2.0, max=2.0), num_dimensions=2)
    #pos_in = nest.spatial.free(pos=nest.random.uniform(min=-2.0, max=2.0), num_dimensions=2)

    nodes_ex = nest.Create("iaf_psc_delta", NE, params=neuron_params, positions=pos_ex)
    nodes_in = nest.Create("iaf_psc_delta", NI, params=neuron_params, positions=pos_ex)
    nest.SetKernelStatus({"rng_seed": random_seed})
    noise = nest.Create("poisson_generator", params={"rate": p_rate})
    espikes = nest.Create("spike_recorder")
    ispikes = nest.Create("spike_recorder")

    nest.CopyModel("static_synapse", "excitatory", {"weight": J_ex, "delay": delay})
    nest.CopyModel("static_synapse", "inhibitory", {"weight": J_in, "delay": delay})

    nest.Connect(noise, nodes_ex, syn_spec="excitatory")
    nest.Connect(noise, nodes_in, syn_spec="excitatory")

    nest.Connect(nodes_ex[:NE], espikes, syn_spec="excitatory")
    nest.Connect(nodes_in[:NI], ispikes, syn_spec="excitatory")

    print("Connecting network")

    print("Excitatory connections")
    p_exc=1.0
    conn_params_ex = {"rule": "pairwise_bernoulli", "p": p_exc, "mask": {"circular": {"radius": 1.5}}}
    nest.Connect(nodes_ex, nodes_ex, conn_params_ex, "excitatory")
    nest.Connect(nodes_ex, nodes_in, conn_params_ex, "excitatory")

    print("Inhibitory connections")
    p_inh=1.0
    conn_params_in = {"rule": "pairwise_bernoulli", "p": p_inh, "mask": {"circular": {"radius": 1.5}}}
    nest.Connect(nodes_in, nodes_ex, conn_params_in, "inhibitory")
    nest.Connect(nodes_in, nodes_in, conn_params_in, "inhibitory")

    fig1 = nest.PlotLayer(nodes_ex, nodesize=50)

    ctr = nest.FindCenterElement(nodes_ex)
    nest.PlotTargets(
        ctr,
        nodes_ex,
        fig=fig1,
        mask=conn_params_ex["mask"],
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


    fig2 = nest.PlotLayer(nodes_ex)
    nest.PlotSources(
        nodes_ex,
        ctr,
        fig=fig2,
        mask=conn_params_ex["mask"],
        src_size=50,
        src_color='black',
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
    target_ids_exc = nest.GetTargetNodes(ctr, nodes_ex)[0]
    target_ids_inh = nest.GetTargetNodes(ctr, nodes_in)[0]

    nest.GetLocalNodeCollection(ctr)

    src_id = int(ctr.tolist()[0])
    print("src_id:", src_id)

    targets_exc = target_ids_exc.tolist()
    targets_inh = target_ids_inh.tolist()
    print(targets_exc)
    print(targets_inh)

    amplitude=160.0
    stim_params = {"amplitude": amplitude, "start": 5150.0, "stop": 18150.0}
    stimulator = nest.Create("dc_generator", params=stim_params)

    # Connect the stimulator to the neuron
    exc_id1 = int(targets_exc[0])
    inh_id1 = int(targets_inh[0])
    inh_id2 = int(targets_inh[1])
    print("inh_id1=", inh_id1)
    print("inh_id2=", inh_id2)
    nest.Connect(stimulator, nodes_ex[src_id-1])
    #nest.Connect(stimulator, nodes_ex[exc_id1-1])
    #nest.Connect(stimulator, nodes_in[inh_id1-NE-1])
    #nest.Connect(stimulator, nodes_in[inh_id2-NE-1])

    endbuild = time.time()

    print("Simulating")

    nest.Simulate(simtime)

    endsimulate = time.time()

    events_ex = espikes.n_events
    events_in = ispikes.n_events

    rate_ex = events_ex / simtime * 1000.0 / NE
    rate_in = events_in / simtime * 1000.0 / NI

    num_synapses_ex = nest.GetDefaults("excitatory")["num_connections"]
    num_synapses_in = nest.GetDefaults("inhibitory")["num_connections"]
    num_synapses = num_synapses_ex + num_synapses_in

    build_time = endbuild - startbuild
    sim_time = endsimulate - endbuild
    #cdict = {"rule": "pairwise_bernoulli", "p": 0.5, "mask": {"circular": {"radius": 0.5}}}

    #nest.Connect(a, b, conn_spec=cdict, syn_spec={"weight": nest.random.uniform(0.5, 2.0)})

    sr1_spikes = espikes.events['senders']
    sr1_times = espikes.events['times']
    sr2_spikes = ispikes.events['senders']
    sr2_times = ispikes.events['times']

    plt.figure('Figure 3')
    ax = plt.subplot()
    ax.plot(sr1_times, sr1_spikes,
            '.', markersize=1, color='blue', label='Exc')
    ax.plot(sr2_times, sr2_spikes,
            '.', markersize=1, color='orange', label='Inh')
    ax.set_xlabel('time (ms)')
    ax.set_ylabel('neuron id')
    ax.legend()


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


    bin_width = 200.0

    plt.figure('Figure 4')
    # Create a histogram plot for excitatory neurons
    plt.hist(spike_times_exc[src_id-1], bins=int(simtime/bin_width), edgecolor='black')  # Adjust the number of bins as needed
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("CV of Excitatory Neurons")

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

    plt.figure('Figure 5')
    plt.plot(bin_centers, hist_counts_all_exc.T.mean(axis=1))
    #plt.legend(targets_exc)
    plt.title("Stim: Excitatory Neurons (p_exc= {}, p_inh= {})".format(p_exc, p_inh))
    plt.xlabel('Time')
    plt.ylabel('Firing Rate (Hz)')

    elep_spike_times_exc = np.array(elep_spike_times_exc)*ms

    spiketrains = [neo.SpikeTrain(times, t_start=0 * ms, t_stop=20000 * ms) for times in elep_spike_times_exc]

    # Specify the parameters for the instantaneous_rate function
    sampling_period = 1 * ms
    kernel = 'auto'  # You can specify a specific kernel shape if needed
    cutoff = 5.0 

    # Call the instantaneous_rate function for each neuron's spike train
    firing_rates_exc = []
    for spiketrain in spiketrains:
        rate_analog_signal = es.instantaneous_rate(spiketrain, sampling_period, kernel=kernel, cutoff=cutoff)
        firing_rates_exc.append(rate_analog_signal.magnitude)
        print(firing_rates_exc)
        print(np.array(firing_rates_exc).shape)


    # Calculate the average firing rate across neurons for each time step
    average_firing_rate_exc = np.mean(firing_rates_exc, axis=0)
    print(average_firing_rate_exc.shape)
    # Create a time axis based on the first spiketrain's duration
    time_axis = np.linspace(0, 20000, len(average_firing_rate_exc))

    # Plot the average firing rate over time
    plt.plot(time_axis, average_firing_rate_exc, color='blue')
    plt.grid(True)

    plt.figure('Figure 6')
    plt.plot(bin_centers, hist_counts_all_exc[2].T)
    #plt.legend(targets_exc)
    plt.title("Stim: One Excitatory Neuron (p_exc= {}, p_inh= {})".format(p_exc, p_inh))
    plt.xlabel('Time')
    plt.ylabel('Firing Rate (Hz)')

    # Plot the average firing rate over time
    plt.plot(time_axis, firing_rates_exc[2], color='blue')
    plt.grid(True)






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

    plt.figure('Figure 7')
    plt.plot(bin_centers, hist_counts_all_inh.T.mean(axis=1))
    #plt.legend(targets_inh)
    plt.title("Stim: Inhibitory Neurons (p_exc= {}, p_inh= {})".format(p_exc, p_inh))
    plt.xlabel('Time')
    plt.ylabel('Firing Rate (Hz)')

    elep_spike_times_inh = np.array(elep_spike_times_inh)*ms

    spiketrains = [neo.SpikeTrain(times, t_start=0 * ms, t_stop=20000 * ms) for times in elep_spike_times_inh]

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
    average_firing_rate_inh = np.mean(firing_rates_inh, axis=0)
    print(average_firing_rate_inh.shape)
    # Create a time axis based on the first spiketrain's duration
    time_axis = np.linspace(0, 20000, len(average_firing_rate_inh))

    # Plot the average firing rate over time
    plt.plot(time_axis, average_firing_rate_inh, color='blue')
    plt.grid(True)

    plt.figure('Figure 8')
    plt.plot(bin_centers, hist_counts_all_inh[2].T)
    #plt.legend(targets_exc)
    plt.title("Stim: One Inhibitory Neuron (p_exc= {}, p_inh= {})".format(p_exc, p_inh))
    plt.xlabel('Time')
    plt.ylabel('Firing Rate (Hz)')

    # Plot the average firing rate over time
    plt.plot(time_axis, firing_rates_inh[2], color='blue')
    plt.grid(True)



    # Distant: 2 edge
    # target ids start from 1
    second_degree_target_exc = nest.GetTargetNodes(nodes_ex[targets_exc[0]-1], nodes_ex)[0]
    second_degree_target_inh = nest.GetTargetNodes(nodes_ex[targets_exc[0]-1], nodes_in)[0]

    second_degree_target_exc = second_degree_target_exc.tolist()
    second_degree_target_inh = second_degree_target_inh.tolist()
    print('second_degree_target_exc:', second_degree_target_exc)
    print('second_degree_target_inh:', second_degree_target_inh)

    data_exc = spike_times_exc[324-1]
    num_bins = int(simtime/bin_width)
    hist_counts, hist_edges = np.histogram(data_exc, bins=num_bins, range=(min(data), max(data)))
    bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2

    plt.figure('Figure 9')
    plt.plot(bin_centers, hist_counts.T/0.2)
    #plt.legend(targets_inh)
    plt.title("Stim: Second distant Excitatory neuron (p_exc= {}, p_inh= {})".format(p_exc, p_inh))
    plt.xlabel('Time')
    plt.ylabel('Firing Rate (Hz)')

    data_exc = np.array(data_exc)*ms

    spiketrain_sec_distant_exc = neo.SpikeTrain(data_exc, t_start=0 * ms, t_stop=20000 * ms)

    # Specify the parameters for the instantaneous_rate function
    sampling_period = 1 * ms
    kernel = 'auto'  # You can specify a specific kernel shape if needed
    cutoff = 5.0 

    # Call the instantaneous_rate function for each neuron's spike train
    rate_analog_signal = es.instantaneous_rate(spiketrain_sec_distant_exc, sampling_period, kernel=kernel, cutoff=cutoff)


    # Extract the firing rate values and time axis
    firing_rate_sec_exc = rate_analog_signal.magnitude
    # Create a time axis based on the first spiketrain's duration
    time_axis = np.linspace(0, 20000, len(firing_rate_sec_exc))

    # Plot the average firing rate over time
    plt.plot(time_axis, firing_rate_sec_exc, color='blue')
    plt.grid(True)

    data_inh = spike_times_inh[461-NE-1]
    num_bins = int(simtime/bin_width)
    hist_counts, hist_edges = np.histogram(data_inh, bins=num_bins, range=(min(data), max(data)))
    bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2

    plt.figure('Figure 10')
    plt.plot(bin_centers, hist_counts.T/0.2)
    #plt.legend(targets_inh)
    plt.title("Stim: Second distant Inhibitory neuron (p_exc= {}, p_inh= {})".format(p_exc, p_inh))
    plt.xlabel('Time')
    plt.ylabel('Firing Rate (Hz)')

    data_inh = np.array(data_inh)*ms

    spiketrain_sec_distant_inh = neo.SpikeTrain(data_inh, t_start=0 * ms, t_stop=20000 * ms)

    # Specify the parameters for the instantaneous_rate function
    sampling_period = 1 * ms
    kernel = 'auto'  # You can specify a specific kernel shape if needed
    cutoff = 5.0 

    # Call the instantaneous_rate function for each neuron's spike train
    rate_analog_signal = es.instantaneous_rate(spiketrain_sec_distant_inh, sampling_period, kernel=kernel, cutoff=cutoff)


    # Extract the firing rate values and time axis
    firing_rate_sec_inh = rate_analog_signal.magnitude
    # Create a time axis based on the first spiketrain's duration
    time_axis = np.linspace(0, 20000, len(firing_rate_sec_inh))

    # Plot the average firing rate over time
    plt.plot(time_axis, firing_rate_sec_inh, color='blue')
    plt.grid(True)






    # The one connected to inhibitory neuron
    # target ids start from 1
    second_degree_target_exc_inh = nest.GetTargetNodes(nodes_in[targets_inh[0]-NE-1], nodes_ex)[0]
    second_degree_target_inh_inh = nest.GetTargetNodes(nodes_in[targets_inh[0]-NE-1], nodes_in)[0]

    second_degree_target_exc_inh = second_degree_target_exc_inh.tolist()
    second_degree_target_inh_inh = second_degree_target_inh_inh.tolist()
    print('second_degree_target_exc:', second_degree_target_exc_inh)
    print('second_degree_target_inh:', second_degree_target_inh_inh)

    data_exc_inh = spike_times_exc[236-1]
    num_bins = int(simtime/bin_width)
    hist_counts, hist_edges = np.histogram(data_exc_inh, bins=num_bins, range=(min(data), max(data)))
    bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2

    plt.figure('Figure 11')
    plt.plot(bin_centers, hist_counts.T/0.2)
    #plt.legend(targets_inh)
    plt.title("Stim: Second distant Excitatory neuron (p_exc= {}, p_inh= {})".format(p_exc, p_inh))
    plt.xlabel('Time')
    plt.ylabel('Firing Rate (Hz)')

    data_exc_inh = np.array(data_exc_inh)*ms

    spiketrain_sec_distant_exc_inh = neo.SpikeTrain(data_exc_inh, t_start=0 * ms, t_stop=20000 * ms)

    # Specify the parameters for the instantaneous_rate function
    sampling_period = 1 * ms
    kernel = 'auto'  # You can specify a specific kernel shape if needed
    cutoff = 5.0 

    # Call the instantaneous_rate function for each neuron's spike train
    rate_analog_signal = es.instantaneous_rate(spiketrain_sec_distant_exc_inh, sampling_period, kernel=kernel, cutoff=cutoff)


    # Extract the firing rate values and time axis
    firing_rate_sec_exc_inh = rate_analog_signal.magnitude
    # Create a time axis based on the first spiketrain's duration
    time_axis = np.linspace(0, 20000, len(firing_rate_sec_exc_inh))

    # Plot the average firing rate over time
    plt.plot(time_axis, firing_rate_sec_exc_inh, color='blue')
    plt.grid(True)

    data_inh_inh = spike_times_inh[466-NE-1]
    num_bins = int(simtime/bin_width)
    hist_counts, hist_edges = np.histogram(data_inh_inh, bins=num_bins, range=(min(data), max(data)))
    bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2

    plt.figure('Figure 12')
    plt.plot(bin_centers, hist_counts.T/0.2)
    #plt.legend(targets_inh)
    plt.title("Stim: Second distant Inhibitory neuron (p_exc= {}, p_inh= {})".format(p_exc, p_inh))
    plt.xlabel('Time')
    plt.ylabel('Firing Rate (Hz)')

    data_inh_inh = np.array(data_inh_inh)*ms

    spiketrain_sec_distant_inh_inh = neo.SpikeTrain(data_inh_inh, t_start=0 * ms, t_stop=20000 * ms)

    # Specify the parameters for the instantaneous_rate function
    sampling_period = 1 * ms
    kernel = 'auto'  # You can specify a specific kernel shape if needed
    cutoff = 5.0 

    # Call the instantaneous_rate function for each neuron's spike train
    rate_analog_signal = es.instantaneous_rate(spiketrain_sec_distant_inh_inh, sampling_period, kernel=kernel, cutoff=cutoff)


    # Extract the firing rate values and time axis
    firing_rate_sec_inh_inh = rate_analog_signal.magnitude
    # Create a time axis based on the first spiketrain's duration
    time_axis = np.linspace(0, 20000, len(firing_rate_sec_inh_inh))

    # Plot the average firing rate over time
    plt.plot(time_axis, firing_rate_sec_inh_inh, color='blue')
    plt.grid(True)


    return np.array(firing_rates_exc), bin_centers, hist_counts_all_exc.T.mean(axis=1)


ff_1, bin_centers, hist_mean_1 = run_sim(1*123)
ff_2, bin_centers, hist_mean_2 = run_sim(2*123)
ff_3, bin_centers, hist_mean_3 = run_sim(3*123)
ff_4, bin_centers, hist_mean_4 = run_sim(4*123)
ff_5, bin_centers, hist_mean_5 = run_sim(5*123)
ff_6, bin_centers, hist_mean_6 = run_sim(6*123)
ff_7, bin_centers, hist_mean_7 = run_sim(7*123)
ff_8, bin_centers, hist_mean_8 = run_sim(8*123)
ff_9, bin_centers, hist_mean_9 = run_sim(9*123)
#upp_new = np.hstack((ff_1, ff_2, ff_3))
#hist_new = np.column_stack((hist_mean_1, hist_mean_2, hist_mean_3))
upp_new = np.hstack((ff_1, ff_2, ff_3, ff_4, ff_5, ff_6, ff_7, ff_8, ff_9))
hist_new = np.column_stack((hist_mean_1, hist_mean_2, hist_mean_3, hist_mean_4, hist_mean_5, hist_mean_6, hist_mean_7, hist_mean_8, hist_mean_9))
mean_hist = hist_new.mean(axis=1)
print(type(hist_new.mean(axis=1)))
print(hist_new.mean(axis=1).shape)
print(ff_2.shape)
print(ff_3.shape)

# Define the smoothing kernel (e.g., Gaussian kernel)
kernel_size = 5  # Adjust the size of the kernel as needed
sigma = 1.0      # Adjust the sigma parameter for Gaussian distribution
kernel = np.exp(-(np.arange(-kernel_size//2, kernel_size//2 + 1)**2) / (2*sigma**2))
kernel /= np.sum(kernel)  # Normalize the kernel so that the sum is 1

# Apply zero-padding to the data
padded_data = np.pad(mean_hist, (kernel_size//2, kernel_size//2), mode='edge')

# Apply convolution to smooth the padded data
smoothed_padded_data = convolve(padded_data, kernel, mode='valid')

# The size of 'smoothed_padded_data' is smaller than the original 'data' due to valid convolution
# If you want to keep the size the same, you can add zero-padding to the smoothed result

# Add zero-padding to the smoothed data to match the original size
padding = (kernel_size - 1) // 2
smoothed_data = np.pad(smoothed_padded_data, (padding, padding), mode='constant')
smoothed_data = smoothed_data[2:-2]
print(smoothed_data)
print(smoothed_data.shape)
#print(upp_new.mean(axis=1).shape)
## Plot the average firing rate over time
plt.figure('Figure 20')
plt.plot(np.linspace(0, 20000, len(smoothed_data)), smoothed_data, color='blue')
plt.plot(bin_centers[:len(smoothed_data)-1], mean_hist[:len(smoothed_data)-1])
plt.grid(True)

plt.show()
plt.close()

