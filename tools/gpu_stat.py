import py3nvml
free_gpus = py3nvml.get_free_gpus()

def get_free_gpu():
    if not True in free_gpus:
        raise Exception
    else:
        return free_gpus.index(True)

