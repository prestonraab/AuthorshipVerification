import os
import time

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Define constants
FINAL_VECTOR_SIZE = 256
NUM_EPOCHS = 10
BATCH_SIZE = 200
LEARNING_RATE = 1e-3
GAMMA = 0.7
HIDDEN_SIZE = 600
DROPOUT = 0.1
LOG_INTERVAL = 5
N_FFT = 512
FFT_OUT = N_FFT // 2 + 1
SAVE_MODEL = True
SAVE_INTERVAL = 5
TEST_INTERVAL = 4
LOAD_DATASETS_FROM_FILE = True
LOAD_MODEL = False
NUM_WORKERS = 2
DO_TESTING = True


def report_time(start_time):
    print(f"Time elapsed: {time.time() - start_time:.4f}")
    return time.time()
