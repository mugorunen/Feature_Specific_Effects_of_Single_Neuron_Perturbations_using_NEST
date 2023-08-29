import numpy as np
import elephant.statistics as es
from quantities import ms
import neo
from scipy.signal import convolve2d
from scipy.signal import convolve
import nest

class NetworkAnalyzer:
    def __init__(self, NE, NI, N_neurons, simtime, bin_width):
        self.NE = NE
        self.NI = NI
        self.N_neurons = N_neurons
        self.simtime = simtime
        self.bin_width = bin_width

    def calculate_avg_firing(self, pop, sim_time, spikes, times, flag):
        if flag == 0:
            neuron_ids = [i for i in range(1, self.NE + 1, 1)]
        else:
            neuron_ids = [i for i in range(1 + self.NE, self.NI + 1 + self.NE, 1)]
        firing_rates = []
        spike_times = []
        for neuron_id in neuron_ids:
            spikes_for_neuron = np.sum(spikes == neuron_id)
            firing_rate = spikes_for_neuron / (sim_time / 1000)  # Convert to Hz
            firing_rates.append(firing_rate)
            time_indices = np.where(spikes == neuron_id)[0]
            time_values = times[time_indices]
            spike_times.append(time_values)
        average_firing_rate = np.mean(firing_rates)
        return average_firing_rate, spike_times

    def calculate_CoV(self, spike_times):
        consecutive_diffs = [np.diff(sublist) for sublist in spike_times]
        mean_isi = [np.mean(sublist) for sublist in consecutive_diffs]
        std_isi = [np.std(sublist) for sublist in consecutive_diffs]
        CoV = [a / b for a, b in zip(std_isi, mean_isi)]
        return CoV

    def create_connectivity(self, nodes_ex, nodes_in):
        connectivity = np.zeros((self.N_neurons, self.N_neurons))
        conn_ex = nest.GetConnections(nodes_ex)
        conn_ex_source = nest.GetStatus(conn_ex, keys='source')
        conn_ex_target = nest.GetStatus(conn_ex, keys='target')
        conn_ex_weight = nest.GetStatus(conn_ex, keys='weight')
        conn_in = nest.GetConnections(nodes_in)
        conn_in_source = nest.GetStatus(conn_in, keys='source')
        conn_in_target = nest.GetStatus(conn_in, keys='target')
        conn_in_weight = nest.GetStatus(conn_in, keys='weight')

        for i in range(len(conn_ex_source)):
            if conn_ex_source[i] <= self.N_neurons and conn_ex_target[i] <= self.N_neurons:
                connectivity[conn_ex_source[i] - 1, conn_ex_target[i] - 1] = conn_ex_weight[i]
        for i in range(len(conn_in_source)):
            if conn_in_source[i] <= self.N_neurons and conn_in_target[i] <= self.N_neurons:
                connectivity[conn_in_source[i] - 1, conn_in_target[i] - 1] = conn_in_weight[i]

        connectivity_matrix = connectivity.T
        return connectivity_matrix
    
    def calculating_firing_rates(self, targets, src_id, spike_times, neuron_type):
         # Plot the mean firing rate of the Excitatory Neurons Connected to the Perturbed One
        hist_counts_all = []
        elep_spike_times=[]
        #targets_exc.remove(src_id)
        for i in targets:
            if i==src_id and neuron_type==0:
                data = spike_times[i-1]
            elif neuron_type==0:
                data = spike_times[i-1]
                elep_spike_times.append(data)
                num_bins = int(self.simtime/self.bin_width)
                hist_counts, hist_edges = np.histogram(data, bins=num_bins, range=(min(data), max(data)))
                hist_counts_all.append(hist_counts/(self.bin_width/1000))
            elif neuron_type==1:
                data = spike_times[i-self.NE-1]
                elep_spike_times.append(data)
                num_bins = int(self.simtime/self.bin_width)
                hist_counts, hist_edges = np.histogram(data, bins=num_bins, range=(min(data), max(data)))
                hist_counts_all.append(hist_counts/(self.bin_width/1000))

        bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
        hist_counts_all = np.array(hist_counts_all)
        avg_hist_counts = hist_counts_all.T.mean(axis=1)

        # Plot the mean firing rate of the Excitatory Neurons Connected to the Perturbed One Using Smoothing Kernel
        elep_spike_times = np.array(elep_spike_times)*ms
        spiketrains = [neo.SpikeTrain(times, t_start=0 * ms, t_stop=self.simtime * ms) for times in elep_spike_times]

        # Specify the parameters for the instantaneous_rate function
        sampling_period = 1 * ms
        kernel = 'auto'  # You can specify a specific kernel shape if needed
        cutoff = 5.0 

        # Call the instantaneous_rate function for each neuron's spike train
        firing_rates_smoothed = []
        for spiketrain in spiketrains:
            rate_analog_signal = es.instantaneous_rate(spiketrain, sampling_period, kernel=kernel, cutoff=cutoff)
            firing_rates_smoothed.append(rate_analog_signal.magnitude)

        # Calculate the average firing rate across neurons for each time step
        average_firing_rate = np.mean(firing_rates_smoothed, axis=0)


        return hist_counts_all, bin_centers, avg_hist_counts, average_firing_rate, firing_rates_smoothed
    
    def smoothing_kernel(self, mean_hist):
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
        
        return smoothed_data


