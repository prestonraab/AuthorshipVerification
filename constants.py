import os
import time

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Define constants
FINAL_VECTOR_SIZE = 128
NUM_EPOCHS = 2000
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
GAMMA = 0.7
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.1
LOG_INTERVAL = 10
N_FFT = 512
FFT_OUT = N_FFT // 2 + 1
SAVE_MODEL = True
LOAD_DATASETS_FROM_FILE = True
LOAD_MODEL = False


def report_time(start_time):
    print(f"Time elapsed: {time.time() - start_time:.4f}")
    return time.time()
