import numpy as np
import elephant.statistics as es
from quantities import ms
import neo
from scipy.signal import convolve2d
from scipy.signal import convolve
import nest

# The file to analyze the activity of the neurons
class NetworkAnalyzer:
    # Initialization of the values
    def __init__(self, NE, NI, N_neurons, simtime, bin_width):
        self.NE = NE
        self.NI = NI
        self.N_neurons = N_neurons
        self.simtime = simtime
        self.bin_width = bin_width

    # Calculation of average firing rate
    def calculate_avg_firing_target(self, pop, target_ids, sim_time, spikes, times, flag):
        
        firing_rates = []
        spike_times = []
        for neuron_id in target_ids:
            spikes_for_neuron = np.sum(spikes == neuron_id)
            firing_rate = spikes_for_neuron / (sim_time / 1000)  # Convert to Hz
            firing_rates.append(firing_rate)
            time_indices = np.where(spikes == neuron_id)[0]
            time_values = times[time_indices]
            spike_times.append(time_values)
        average_firing_rate = np.mean(firing_rates)
        return average_firing_rate, spike_times

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

    # Calculate coefficient of variance
    def calculate_CoV(self, spike_times):
        consecutive_diffs = [np.diff(sublist) for sublist in spike_times]
        mean_isi = [np.mean(sublist) for sublist in consecutive_diffs]
        std_isi = [np.std(sublist) for sublist in consecutive_diffs]
        CoV = [a / b for a, b in zip(std_isi, mean_isi)]
        return CoV


    # Create connecticity according to results. Not being used in the main code
    def create_connectivity(self, nodes_ex, nodes_in):
        connectivity_target = np.zeros((self.N_neurons, self.N_neurons))
        connectivity_source = np.zeros((self.N_neurons, self.N_neurons))
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
                connectivity_target[conn_ex_source[i] - 1, conn_ex_target[i] - 1] = conn_ex_weight[i]
                connectivity_source[conn_ex_target[i] - 1, conn_ex_source[i] - 1] = conn_ex_weight[i]
        for i in range(len(conn_in_source)):
            if conn_in_source[i] <= self.N_neurons and conn_in_target[i] <= self.N_neurons:
                connectivity_target[conn_in_source[i] - 1, conn_in_target[i] - 1] = conn_in_weight[i]
                connectivity_source[conn_in_target[i] - 1, conn_in_source[i] - 1] = conn_in_weight[i]

        connectivity_target_matrix = connectivity_target.T
        connectivity_source_matrix = connectivity_source.T
        return connectivity_target_matrix, connectivity_source_matrix
    
    # Calculate firing rate for each neuron
    def calculating_firing_rates(self, src_id, spike_times, neuron_type):
         # Plot the mean firing rate of the Excitatory Neurons Connected to the Perturbed One
        hist_counts_all = []
        elep_spike_times=[]
        if neuron_type==0:
            num = self.NE
        else:
            num = self.NI
        #targets_exc.remove(src_id)
        for i in range(num):
            if np.any(i+1 == src_id):
                data = spike_times[i]
            elif neuron_type==0:
                data = spike_times[i]
                elep_spike_times.append(data)
                num_bins = int(self.simtime/self.bin_width)
                hist_counts, hist_edges = np.histogram(data, bins=num_bins, range=(min(data), max(data)))
                hist_counts_all.append(hist_counts/(self.bin_width/1000))
            elif neuron_type==1:
                data = spike_times[i]
                elep_spike_times.append(data)
                num_bins = int(self.simtime/self.bin_width)
                hist_counts, hist_edges = np.histogram(data, bins=num_bins, range=(min(data), max(data)))
                hist_counts_all.append(hist_counts/(self.bin_width/1000))

        bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
        hist_counts_all = np.array(hist_counts_all)
        avg_hist_counts = hist_counts_all.T.mean(axis=1)


        return hist_counts_all, bin_centers, avg_hist_counts
    

    # Smoothing kernel to better visualize the data
    def smoothing_kernel(self, mean_hist):
        # Define the smoothing kernel (e.g., Gaussian kernel)
        kernel_size = 40  # Adjust the size of the kernel as needed
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
        #padding = (kernel_size - 1) // 2
        #smoothed_data = np.pad(smoothed_padded_data, (padding, padding), mode='constant')
        
        return smoothed_padded_data
    
    # Find the target of the neuron
    def find_src_target_ids(self, nodes_ex, nodes_in):
        ctr = nest.FindCenterElement(nodes_ex)
        src_id = int(ctr.tolist()[0])
        print("src_id:", ctr.tolist())

        target_ids_exc = nest.GetTargetNodes(ctr, nodes_ex)[0]
        target_ids_inh = nest.GetTargetNodes(ctr, nodes_in)[0]
        targets_exc = target_ids_exc.tolist()
        targets_inh = target_ids_inh.tolist()

        return ctr, src_id, targets_exc, targets_inh
    
    # Find the neurons that do not have direct connection to perturbed one
    def get_nodes_2_nodes_away(self, ctr, neuron_id, targets, nodes_ex, nodes_in, neuron_type):

        if neuron_type==0:
            target_2_exc = nest.GetTargetNodes(nodes_ex[neuron_id-1], nodes_ex)[0]
            target_2_inh = nest.GetTargetNodes(nodes_ex[neuron_id-1], nodes_in)[0]
        else:
            target_2_exc = nest.GetTargetNodes(nodes_in[neuron_id-self.NE-1], nodes_ex)[0]
            target_2_inh = nest.GetTargetNodes(nodes_in[neuron_id-self.NE-1], nodes_in)[0]


    
        return target_2_exc.tolist(), target_2_inh.tolist()


