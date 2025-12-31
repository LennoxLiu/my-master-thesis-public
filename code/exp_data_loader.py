from scipy.io import loadmat
import numpy as np

# no normalization across neurons
# return bc_data, pom_data
def load_grouped_data(data_path='./data/testFile.mat'):
    # load data from mat file
    data = loadmat(data_path)
    sortedData=data['sortedData']
    neuronIDs=sortedData[:,0]
    neuronIDs=[neuronIDs[i][0] for i in range(len(neuronIDs))]
    # print(neuronIDs)
    # record all index that are in 'BC' and 'POm'
    BC_index=[]
    POm_index=[]
    for i in range(len(sortedData)):
        if sortedData[i][3]=='BC': # and sortedData[i][0] == '1' or sortedData[i][0] == '37':
            BC_index.append(i)
        if sortedData[i][3]=='POm': # and sortedData[i][0] == '133' or sortedData[i][0] == '142':
            POm_index.append(i)

    # convert sortedData to table of histograms
    bc_data = []
    for bc_id in BC_index:
        bc = np.array(sortedData[bc_id][1], dtype=np.float64).flatten()
        bc_data.append(bc/1000)  # convert to seconds

    pom_data = []
    for pom_id in POm_index:
        pom = np.array(sortedData[pom_id][1], dtype=np.float64).flatten()
        pom_data.append(pom/1000)  # convert to seconds

    return bc_data, pom_data

def get_list_by_length_criteria(list_of_lists, criteria):
    """
    Finds the shortest, longest, or middle-length array from a list of arrays.
    
    :param list_of_lists: A list where each element is an array/list.
    :param criteria: 'shortest', 'longest', or 'middle'.
    :return: The array matching the criteria, or None if the list is empty.
    """
    if not list_of_lists:
        return None

    # 1. Sort the list of arrays based on their length
    # This places arrays from shortest to longest.
    sorted_lists = sorted(list_of_lists, key=len)
    n = len(sorted_lists)

    if criteria == 'shortest':
        # The shortest is the first element after sorting
        return sorted_lists[0]
    elif criteria == 'longest':
        # The longest is the last element after sorting
        return sorted_lists[-1]
    elif criteria == 'middle':
        # The middle array is at index (n - 1) // 2 
        # This works for both odd (true middle) and even (shorter of the two central) lengths
        middle_index = (n - 1) // 2
        return sorted_lists[middle_index]
    else:
        raise ValueError("Criteria must be 'shortest', 'longest', or 'middle'")
    
if __name__ == "__main__":
    # Example usage
    bc_data, pom_data = load_grouped_data()
    print(f"Loaded {len(bc_data)} BC neurons and {len(pom_data)} POm neurons.")
    print(pom_data[0][-20:])  # Print spike times of the first POm neuron