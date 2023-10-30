import os
import pickle
import scipy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class NinaProSubject:
    def __init__(self, subject: int):
        """
        Initialize NinaProSubject with a subject identifier.

        Args:
            subject (int): Subject identifier.
        """
        self.subject = subject
        self.emg = None
        self.labels = None
        self.indices = []

    def load_db(self, path: str):
        """
        Load data from a NinaPro database file.

        Args:
            path (str): Path to the database directory.
        """
        data = scipy.io.loadmat(os.path.join(path, f"DB2_s{self.subject}/S{self.subject}_E1_A1.mat"))
        self.emg = data['emg']
        self.labels = data['restimulus']
        self.compute_indices()

    def compute_indices(self):
        """
        Compute indices based on label differences.
        """

        # shift labels by 1 and substract from labels
        # this will create a nonzero entry whenever a change of value happens
        # The non-zero indices correspond to the edges of the intervals
        labels_rolled = np.roll(np.copy(self.labels), 1)
        delta = (self.labels - labels_rolled)
        self.indices = np.where(delta != 0)[0]

    def save_object(self, savepath: str):
        """
        Save the NinaProSubject object to a pickle file.

        Args:
            savepath (str): Path to the save directory.
        """
        with open(os.path.join(savepath, f"sub_{self.subject}.pkl"), "wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_object(subject: int, path: str):
        """
        Load a NinaProSubject object from a pickle file.

        Args:
            subject (int): Subject identifier.
            path (str): Path to the directory containing the pickle file.

        Returns:
            NinaProSubject: Loaded NinaProSubject object.
        """
        with open(os.path.join(path, f"sub_{subject}.pkl"), "rb") as file:
            obj = pickle.load(file)
        return obj


def create_db(source_path: str, target_path: str):
    """
    Create a database of NinaProSubject objects by loading data from source files
    and saving them to target files.

    Args:
        source_path (str): Path to the source directory containing NinaPro data.
        target_path (str): Path to the target directory to save NinaProSubject objects.
    """
    print('Saving subject objs. to ' + target_path + '...')
    for i in tqdm(range(0, 40)):
        subject_number = i + 1
        sub = NinaProSubject(subject_number)
        sub.load_db(source_path)
        sub.save_object(target_path)


#create_db('/Users/dc23/Desktop/datasets/Ninapro/db2', '/Users/dc23/Desktop/datasets/Ninapro/db2-preprocessed')
