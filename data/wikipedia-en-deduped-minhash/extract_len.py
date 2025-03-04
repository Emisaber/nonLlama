import os
import pickle
import numpy as np

data_dir = os.path.dirname(__file__)

train_data = np.memmap(os.path.join(data_dir, "train.bin"),
                        dtype=np.uint16, mode='r')

val_data = np.memmap(os.path.join(data_dir, f"val.bin"),
                        dtype=np.uint16, mode='r')

train_data_len = len(train_data)
val_data_len = len(val_data)

meta = {
    'train_data_len': train_data_len,
    'val_data_len': val_data_len,
}

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)