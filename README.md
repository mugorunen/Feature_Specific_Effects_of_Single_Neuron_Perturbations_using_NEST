# Modeling Feature-specific Spiking Neural Network Responses to Optogenetic Perturbations
In this project, we analyzed the effects of single-neuron perturbation using the NEST simulation environment. Chettih et al. [1] investigated the impact of single-neuron perturbations in layer 2/3 of the primary visual cortex (V1) in awake mice in vivo. They utilized two-photon optogenetics to trigger action potentials and calcium imaging to monitor changes in intracellular calcium levels. When a single excitatory neuron was perturbed, the activity of neurons similarly tuned to the perturbed one was suppressed, suggesting feature competition in V1.

Sadeh et al. [2] employed computational modeling of single-neuron perturbations to identify connectivity profiles where this effect is observed. Depending on the dominance values (a: E->I dominance, g: inhibition dominance), the intermediate similarity regime showed feature competition, while the highly similar regime exhibited feature amplification for low or moderate dominance values. They utilized a rate-based network model consisting of 400 excitatory and 400 inhibitory neurons.

In our study, we observed the effect of dominance values using spiking neural networks and successfully replicated some results from the Sadeh paper. The connection weights between neurons were set according to their orientation preference similarity using von Mises distribution. A Poisson input was given to every neuron to achieve a balanced network. In each case, a single neuron was perturbed, and the effects on neighboring neurons were observed.

Examining the figure below, it is evident that to obtain feature-specific suppression, strong inhibition dominance alone is not sufficient. It needs to be combined with strong E->I dominance. For highly similar regimes, feature-specific effects depend on dominance values. If the values are low or moderate, there is feature-specific amplification. If they are high, feature-specific suppression is more prominent. The y-axis represents delta rate, which is equal to the difference in average activity between perturbation and no perturbation cases. The result is also normalized with the average activity in the no perturbation case. x axis shows the orientation preference difference between neurons and perturbed neuron.

![Effect of single neuron perturbation](Figures/figure1)

## How to run the code
One needs to run main.py to start the simulation. network_functions.py contains helper functions to analyze the network activity. plotting_functions.py contains helper functions to visualize the data.

NEST version: 3.6.0
Python version: 3.11.5


## References
1. Chettih, S.N., Harvey, C.D. Single-neuron perturbations reveal feature-specific competition in V1. Nature 567, 334–340 (2019). https://doi.org/10.1038/s41586-019-0997-6

2. Sadeh S, Clopath C. Theory of neuronal perturbome in cortical networks. Proc Natl Acad Sci U S A. 2020 Oct 27;117(43):26966-26976. doi: 10.1073/pnas.2004568117. Epub 2020 Oct 14. PMID: 33055215; PMCID: PMC7604497.
