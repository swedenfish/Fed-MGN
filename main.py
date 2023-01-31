from model import MGN_NET
import config
import scipy
import helper
import torch

list1 = []
MGN_NET.train_model(n_max_epochs = config.n_max_epochs, 
                        data_path = config.DATASET_PATH, 
                        early_stop = config.early_stop,
                        model_name = "MGN_NET",
                        n_folds = config.number_of_folds,
                        fed = False,
                        loss_table_list = list1)

list2 = []
MGN_NET.train_model(n_max_epochs = config.n_max_epochs, 
                        data_path = config.DATASET_PATH, 
                        early_stop = config.early_stop,
                        model_name = "MGN_NET",
                        n_folds = config.number_of_folds,
                        fed = True,
                        loss_table_list = list2)

helper.plotLosses(list1, list2)

# print(list1)
# print(list2)

# a = scipy.io.loadmat("data_nc_asd_L\ASD LH\LHASDSub1.mat")
# # for key in a.keys():
# #     print(key)
# #     print(a[key])
# print(a["views"].shape)

# helper.clear_dir("input")
# helper.load_input_from_dir_of_mats("data_nc_asd_L\ASD LH")