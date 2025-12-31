import matplotlib.pyplot as plt
import numpy as np
from tick.plot import plot_hawkes_kernels
from tick.hawkes import SimuHawkesSumExpKernels, SimuHawkesMulti, \
    HawkesSumExpKern, HawkesConditionalLaw, HawkesEM, SimuHawkes, HawkesKernelSumExp
from tick.plot import plot_point_process, plot_hawkes_kernel_norms
import tick

# adjacency : np.ndarray, shape=(n_nodes, n_nodes, n_decays)
def simulateHawkesProcess(decays, baseline, adjacency, end_time=120, n_realizations=1, seed=1040):
    hawkes_exp_kernels = SimuHawkesSumExpKernels(
        adjacency=adjacency, decays=decays, baseline=baseline,
        end_time=end_time, verbose=False, seed=seed)

    multi = SimuHawkesMulti(hawkes_exp_kernels, n_simulations=n_realizations, n_threads=0)
    multi.simulate()

    return multi


def simulateSingleHawkesProcess(decays, baseline, intensity, end_time=120, seed=1040):
    hawkes_exp_kernels = HawkesKernelSumExp(intensities=intensity, decays=decays)
    print("Simulating using Hawkes Kernel: ", hawkes_exp_kernels)
    print("Baseline: ", baseline)
    single = SimuHawkes(kernels=[[hawkes_exp_kernels]], baseline=baseline,
                        end_time=end_time, verbose=False, seed=seed)
    single.track_intensity(1)
    single.simulate()

    return single, hawkes_exp_kernels

class ConditionalHawkes():
    def __init__(self, decays, baseline, adjacency, end_time=120, seed=1039):
        self.decays = decays
        self.baseline = baseline
        self.adjacency = adjacency
        self.end_time = end_time
        self.seed = seed

        self.simulate_yyx()
        self.fit_yy()

    def simulate_yyx(self):
        print("Simulating Hawkes process with adjacency: ", self.adjacency)
        multi = SimuHawkesSumExpKernels(
        adjacency=self.adjacency, decays=self.decays, baseline=self.baseline,
        end_time=self.end_time, verbose=False, seed=self.seed)

        # track intensity at each event time for entropy calcualtion
        multi.track_intensity(intensity_track_step=1)
        multi.simulate()
        self.timestamps = multi.timestamps
        print("len(tracked_intensity)", len(multi.tracked_intensity[1]))
        # print("intensity_tracked_times", multi.intensity_tracked_times[:10])
        # print("timestamps", multi.timestamps[0][:10], multi.timestamps[1][:10])

        self.multi_hawkes = multi


    # Use a single Hawkes process to fit the first process
    def fit_yy(self):
        fit_decays = [1] # decays to fit
        print("Fitting single Hawkes process on the first process with kernel decays:", fit_decays)
        learner = HawkesSumExpKern(fit_decays, penalty='elasticnet', max_iter=10000 ,tol=1e-8,verbose=True) # have to use penalty='elasticnet' to avoid bad fit
        learner.fit([self.timestamps[0]])
        # fig = plot_hawkes_kernels(learner, hawkes=hawkes_exp_kernels, show=True)

        self.learner = learner
        self.kernels_yy = HawkesKernelSumExp(intensities=learner.adjacency[0][0], decays=self.learner.decays)
    
    # Need to call fit_yy before this
    def simulate_yy(self):
        single = SimuHawkes(kernels=[[self.kernels_yy]], baseline=self.learner.baseline,
                        end_time=self.end_time, verbose=False, seed=self.seed+1)
        
        # track intensity at each event time, NOT for entropy calcualtion
        single.track_intensity(intensity_track_step=1)
        single.simulate()

        self.single_hawkes = single
        learned_timestamps = single.timestamps[0]
        
        return learned_timestamps

    def get_yyx_intensities(self, atol=1e-10, rtol=0):
        print("Getting intensities for the multi (yyx) process")

        tracked_intensity = self.multi_hawkes.tracked_intensity[0]
        intensity_tracked_times = self.multi_hawkes.intensity_tracked_times
        target_timestamps = self.multi_hawkes.timestamps[0]

        matching_indices = []

        i = 0  # Pointer for intensity_tracked_times
        j = 0  # Pointer for target_timestamps

        N = len(intensity_tracked_times)
        M = len(target_timestamps)

        while i < N and j < M:
            tracked_time = intensity_tracked_times[i]
            timestamp = target_timestamps[j]

            # Use numpy.isclose for robust floating-point comparison
            if np.isclose(tracked_time, timestamp, rtol=rtol, atol=atol):
                matching_indices.append(i)
                i += 1
                j += 1
            elif tracked_time < timestamp:
                # If tracked_time is smaller, advance its pointer to find a potentially larger value
                i += 1
            else: # tracked_time > timestamp
                # If tracked_time is larger, advance the timestamp pointer to find a potentially larger value
                j += 1

        matching_indices = np.array(matching_indices)
        chosen_tracked_intensity = tracked_intensity[matching_indices]

        return chosen_tracked_intensity

    # Get intensities for all timestamps in the first process
    def get_yy_intensities(self):
        print("Getting intensities for the single (yy) process")
        intensities = np.zeros(len(self.timestamps[0]))
        for i, t in enumerate(self.timestamps[0]):
            history = self.timestamps[0][self.timestamps[0] < t]
            if len(history) > 0:
                intensity = self.kernels_yy.get_values(t - history)
                intensity = np.sum(intensity, axis=0) + self.learner.baseline[0]
            else:
                intensity = self.learner.baseline[0]
            intensities[i] = intensity
        
        return intensities

    # Differential entropy of the conditional pdf Y_present | Y_history, X_history in nats
    # This condontional pdf is an exponential distribution with rate parameter equal to the intensity,
    # so the entropy has a closed form solution, no need for numerical integration.
    def yyx_pdf_entropy(self):
        intensities = self.get_yyx_intensities()
        return 1 - np.mean(np.log(intensities)) # or np.log2(e) - np.log2(intensities) for bits
    
    def yy_pdf_entropy(self):
        intensities = self.get_yy_intensities()
        return 1 - np.mean(np.log(intensities))

    # nats per event
    def get_te(self):
        # TE(Y_present; Y_history, X_history) = H(Y_present | Y_history) - H(Y_present | Y_history, X_history)
        H_yy = self.yy_pdf_entropy()
        H_yyx = self.yyx_pdf_entropy()
        print(f"h(Y_present; Y_history) = {H_yy} nats")
        print(f"h(Y_present; Y_history, X_history) = {H_yyx} nats")

        return H_yy - H_yyx


if __name__ == '__main__':
    end_time = 60 * 1

    decays = [1]
    baseline = [10, 30]
    adjacency = [[[0], [0]],
                [[0], [0]]]

    # multi_timestamps = simulateHawkesProcess(decays, baseline, adjacency, end_time=end_time, n_realizations=n_realizations, seed=1039).timestamps
    # print("Simulated process 1: ", len(multi_timestamps[0][0]), "events")
    # # learner = HawkesEM(4, kernel_size=10, max_iter=1000, n_threads=0, verbose=True, tol=1e-5)
    # learner = HawkesSumExpKern([0.5, 1, 2, 4], penalty='elasticnet', max_iter=10000 ,tol=1e-8,verbose=True) # have to use penalty='elasticnet' to avoid bad fit
    # learner.fit([multi_timestamps[0][0]])
    # # learner.fit(multi_timestamps)

    # # fig = plot_hawkes_kernels(learner, hawkes=hawkes_exp_kernels, show=True)

    # # # Plot value of intensity for a given realization with the fitted parameters
    # # learner.plot_estimated_intensity([multi_timestamps[0][0]], n_points=1000, t_min=40, t_max=45, intensity_track_step=None, show=True, ax=None)
    # learned_hawkes, kernels = simulateSingleHawkesProcess(learner.decays, learner.baseline, learner.adjacency[0][0],
    #                       end_time=end_time, seed=1040)
    # learned_timestamps = learned_hawkes.timestamps[0]
    # print("Learned process 1: ", len(learned_timestamps), "events")

    # # Plot the learned Hawkes process against the original process in histogram
    
    # print(f"Actual baseline: {baseline}")
    # print(f"Learned baseline: {learner.baseline}")
    # # print(f"Actual adjacency: {adjacency}")
    # # print(f"Learned adjacency: {learner.adjacency}")

    # actual_intervals = np.log10(np.diff(multi_timestamps[0][0]))
    # learned_intervals = np.log10(np.diff(learned_timestamps))
    # fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    # histogram_bins = np.linspace(np.min([np.min(actual_intervals), np.min(learned_intervals)]),
    #                              np.max([np.max(actual_intervals), np.max(learned_intervals)]), 75)
    
    # ax.hist(
    #     actual_intervals, bins=histogram_bins, 
    #     color='darkorange', edgecolor='black', alpha=0.5, label=('Actual(#Events: {})'.format(len(actual_intervals)) )
    # )
    # ax.hist(
    #     learned_intervals, bins=histogram_bins,
    #     color='skyblue', edgecolor='black', alpha=0.5, label=('Learned(#Events: {})'.format(len(learned_intervals)))
    # )
    
    # ax.set_title('Histogram of Events')
    # ax.set_xlabel('Log10 Inter-event Time')
    # ax.set_ylabel('Number of Events')
    # ax.legend()
    # plt.tight_layout()
    # plt.show()

    
    hawkes = ConditionalHawkes(decays=decays, baseline=baseline, adjacency=adjacency, end_time=end_time, seed=1042)
    print("TE(Y_present; Y_history, X_history): ", hawkes.get_te())