#Some important constants
RANDOM_SEED = 35813

#Some important paths
# DATASET_PATH = "./simulated_dataset/example.npy"
DATASET_PATH = "./input/dataset.npy"

MODEL_WEIGHT_BACKUP_PATH = "./output/model_weights"
DEEP_CBT_SAVE_PATH = "./output/deep_cbts"
TEMP_FOLDER = "./temp"

test_func = True

#Control variables
number_of_clients = 3
# fed = False
early_stop_rounds = 4
early_stop_distance = 1
early_stop_interval = 50
number_of_folds = 4

if test_func:
    early_stop_distance = 1
    early_stop_interval = 50 
    number_of_folds = 4

# Add the proximal term or not
prox = False

# Non-Strugglers portion
# The smaller the portion is, the better the fedprox performs
portion = 1

# Split the data in an non-iid way for each client by setting random numbers
non_iid_by_numbers = False

# Split the data in an non-iid way for each client by using clustering
non_iid_by_clustering = True

boardcast_first = True

# For TW. Directly use TW when there exist clients with early stopping, otherwise fed-avg
twavg = False

# take 0.5 of global and 0.5 of client
half_combine = True

# take 1 if global
simplytake_combine = True

# fed-rank
rank = True
# Determines the portion of weight transfer. 1 is default
rank_factor = 1

rankprime = True

# Temporal Weighting
tw = True

# Avg by num of data
fedavg = True

n_max_epochs = 500
early_stop = True

mu = 0.001

# Normalize views when calculating rep loss to avoid overfitting large-number views
weighted_loss = True

# Should be decided by input shape
number_of_views = 6

number_of_parameters = 12

#Model Configuration
N_ROIs = 35
N_RANDOM_SAMPLES = 10

#Model hyperparams
#Following config_real_dataset
Nattr = number_of_views
CONV1 = 36
CONV2 = 24
CONV3 = 8
lambda_kl = 10
# CONV1 = 36
# CONV2 = 24
# CONV3 = 5
# lambda_kl = 25


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