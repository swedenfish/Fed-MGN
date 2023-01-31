from model import MGN_NET
import config
import scipy
import helper
import torch
import numpy as np

# Load dataset from MAT files
# helper.clear_dir("input")
# helper.load_input_from_dir_of_mats("data_nc_asd_L\ASD LH")

loss_list_non_fed = []
MGN_NET.train_model(n_max_epochs = config.n_max_epochs, 
                        data_path = config.DATASET_PATH, 
                        early_stop = config.early_stop,
                        model_name = "MGN_NET",
                        n_folds = config.number_of_folds,
                        fed = False,
                        loss_table_list = loss_list_non_fed)

# Reset all the seeds to make sure identicial conditions
torch.manual_seed(35813)
np.random.seed(35813)
loss_list_fed = []
MGN_NET.train_model(n_max_epochs = config.n_max_epochs, 
                        data_path = config.DATASET_PATH, 
                        early_stop = config.early_stop,
                        model_name = "MGN_NET",
                        n_folds = config.number_of_folds,
                        fed = True,
                        loss_table_list = loss_list_fed)

helper.plotLosses(loss_list_non_fed, loss_list_fed)
