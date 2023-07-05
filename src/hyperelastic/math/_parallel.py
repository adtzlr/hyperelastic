import numpy as np
from multiprocessing import cpu_count

def partition(x, threads):
    F, statevars = x[0], x[-1]
    
    shape = F.shape[2:]
    axis = -(1 + np.argmax(shape[::-1]))
    chunksize = shape[axis] // threads
    
    F_list = np.array_split(F, np.arange(0, chunksize, shape[axis])[1:])
    
    if statevars is not None:
        if statevars is F:
            statevars_list = len(F_list) * [None]
        else:
            if statevars.shape[axis] == shape[axis]:
                statevars_list = np.array_split(
                    statevars, np.arange(0, chunksize, shape[axis])
                )
            elif statevars.shape[axis] == 1:
                statevars_list = len(F_list) * [statevars]
            else:
                raise ValueError("Unable to partition statevars into chunks.")
                
    return [F_list, statevars_list], axis