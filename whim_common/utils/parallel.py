from _ctypes import sizeof
from multiprocessing import sharedctypes
import warnings
from numpy import ctypeslib
import numpy


class SharedMemoryArray(object):
    """
    Simple wrapper to pass a Numpy array between processes in shared memory.

    Instantiate using `from_array()` and then pass the object between processes, getting the
    Numpy array out within the process using `get_array()`.

    It assumes the array is not going to be written to. This could also be handled, by getting a
    Numpy array back in the creating process that wraps the shared memory array, but at the moment
    I don't need it.

    """
    def __init__(self, ctypes_array, shape):
        self.ctypess_array = ctypes_array
        self.shape = shape

    @staticmethod
    def from_array(arr):
        if arr.dtype != numpy.float64:
            raise TypeError("can only currently put float64 arrays in shared memory: got %s" % arr.dtype)

        size = arr.size
        # Keep a note of the original shape
        shape = arr.shape
        # Flatten the array for putting in shared memory
        arr.shape = size

        # Wrap in a shared memory array
        arr_ctypes = sharedctypes.RawArray('d', arr)

        shared_array = SharedMemoryArray(arr_ctypes, shape)
        return shared_array

    def get_array(self):
        # This always produces a warning, because of a Python bug: suppress them
        # (http://stackoverflow.com/questions/4964101/pep-3118-warning-when-using-ctypes-array-as-numpy-array)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Get a Numpy array out that uses the shared memory
            arr = ctypeslib.as_array(self.ctypess_array)
        # Use the original array's shape (had to be flattened for RawArray)
        arr.shape = self.shape
        return arr