# length of key   | 4 bytes                   | int32
# key             | length of key bytes       | char[]
# dtype           | 4 bytes                   | int32 (1: torch.float32, 2: torch.float64)
# length of shape | 4 bytes                   | int32
# shape           | length of shape * 4 bytes | int32[]
# data            | length of data bytes      | float32/float64

import torch
import numpy as np


def get_dtype(dtype : str):
    if dtype == "torch.float32":
        return 1
    elif dtype == "torch.float64":
        return 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def get_item(params, key):
    item : bytes = np.int32(len(key)).tobytes() 
    item += key.encode("ascii")
    item += np.int32(get_dtype(str(params[key].dtype))).tobytes()
    item += np.int32(len(params[key].shape)).tobytes()
    item += np.array(params[key].shape, dtype=np.int32).tobytes()
    item += params[key].numpy().tobytes()
    return item

def save_params_to_file(params, filename):
    with open(filename, "wb") as f:
        for key in params.keys():
            item = get_item(params, key)
            f.write(item)


"""
download model from https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/tree/main
"""
if __name__ == "__main__":
    models = ["gpt2-small-124M.pth", "gpt2-medium-355M.pth", "gpt2-large-774M.pth", "gpt2-xl-1558M.pth"]
    save_params_to_file(torch.load(models[0]), "gpt2-small-124M.bin")
