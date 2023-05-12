#Some important constants
RANDOM_SEED = 35813

#Some important paths
# DATASET_PATH = "./simulated_dataset/example.npy"
DATASET_PATH = "./input/dataset.npy"

MODEL_WEIGHT_BACKUP_PATH = "./output/model_weights"
DEEP_CBT_SAVE_PATH = "./output/deep_cbts"
TEMP_FOLDER = "./temp"

#Control variables
number_of_clients = 3
# fed = False
early_stop_rounds = 10
n_max_epochs = 100
early_stop = True
number_of_folds = 4
update_freq = 1

temporal_weighting = False
average_all = True

# Should be decided by input shape
number_of_views = 6

#Model Configuration
N_ROIs = 35
N_RANDOM_SAMPLES = 10

#Model hyperparams
Nattr = number_of_views
CONV1 = 36
CONV2 = 24
CONV3 = 5
lambda_kl = 25

PARAMS = {
            "learning_rate" : 0.0006,
            "n_attr": Nattr,
            "lambda_kl": lambda_kl,
            "Linear1" : {"in": Nattr, "out": CONV1},
            "conv1": {"in" : 1, "out": CONV1},
            
            "Linear2" : {"in": Nattr, "out": CONV1 * CONV2},
            "conv2": {"in" : CONV1, "out": CONV2},
            
            "Linear3" : {"in": Nattr, "out": CONV2 * CONV3},
            "conv3": {"in" : CONV2, "out": CONV3}, 
        }