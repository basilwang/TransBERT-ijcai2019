import cPickle
from cPickle import PickleError
import sys


def memory_size(obj):
    try:
        return sys.getsizeof(cPickle.dumps(obj))
    except PickleError:
        return None
    except TypeError:
        return None