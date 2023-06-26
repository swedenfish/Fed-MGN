from model import MGN_NET
import config
import scipy
import helper
import torch
import random
import numpy as np
import shutil
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Load dataset from MAT files
helper.clear_dir("input")
# helper.load_input_from_dir_of_mats("data_nc_asd_L/NC LH")
helper.load_input_from_dir_of_mats("data_nc_asd_L/ASD LH")

loss_compare_list = helper.loss_compare_list_init(config.number_of_folds, config.number_of_clients, config.n_max_epochs)
rep_loss_compare_list = helper.loss_compare_list_init(config.number_of_folds, config.number_of_clients, config.n_max_epochs)
kl_loss_compare_list = helper.loss_compare_list_init(config.number_of_folds, config.number_of_clients, config.n_max_epochs)

helper.clear_output()

# The larger the frequency, the harder to converge (and the better the fedprox is)
if config.test_func:
    print("********* With federation *********")
    print("update frequency = 10")
    # Reset all the seeds to make sure identicial conditions
    torch.manual_seed(35813)
    np.random.seed(35813)
    random.seed(35813)
    loss_list_fed1 = []
    MGN_NET.train_model(n_max_epochs = config.n_max_epochs, 
                            data_path = config.DATASET_PATH, 
                            early_stop = config.early_stop,
                            model_name = "MGN_NET",
                            n_folds = config.number_of_folds,
                            fed = True,
                            loss_table_list = loss_list_fed1,
                            loss_vs_epoch = loss_compare_list,
                            rep_vs_epoch = rep_loss_compare_list,
                            kl_vs_epoch = kl_loss_compare_list,
                            update_freq = 10)
    np.save("./output/loss_list_fed1", loss_list_fed1)
    loss_list_fed1 = np.load("./output/loss_list_fed1.npy")
    helper.plotSingleLoss(loss_list_fed1)
    
    # Backup
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    shutil.copytree("./output" , desktop + "/output" + current_time)
    
else:

    print("********* Without federation *********")
    torch.manual_seed(35813)
    np.random.seed(35813)
    random.seed(35813)
    loss_list_non_fed = []
    MGN_NET.train_model(n_max_epochs = config.n_max_epochs, 
                            data_path = config.DATASET_PATH, 
                            early_stop = config.early_stop,
                            model_name = "MGN_NET",
                            n_folds = config.number_of_folds,
                            fed = False,
                            loss_table_list = loss_list_non_fed,
                            loss_vs_epoch = loss_compare_list,
                            rep_vs_epoch = rep_loss_compare_list,
                            kl_vs_epoch = kl_loss_compare_list)

    print("********* With federation *********")
    print("update frequency = 1")
    # Reset all the seeds to make sure identicial conditions
    torch.manual_seed(35813)
    np.random.seed(35813)
    random.seed(35813)
    loss_list_fed1 = []
    MGN_NET.train_model(n_max_epochs = config.n_max_epochs, 
                            data_path = config.DATASET_PATH, 
                            early_stop = config.early_stop,
                            model_name = "MGN_NET",
                            n_folds = config.number_of_folds,
                            fed = True,
                            loss_table_list = loss_list_fed1,
                            loss_vs_epoch = loss_compare_list,
                            rep_vs_epoch = rep_loss_compare_list,
                            kl_vs_epoch = kl_loss_compare_list,
                            update_freq = 1)

    print("********* With federation *********")
    print("update frequency = 10")
    torch.manual_seed(35813)
    np.random.seed(35813)
    random.seed(35813)
    loss_list_fed10 = []
    MGN_NET.train_model(n_max_epochs = config.n_max_epochs, 
                            data_path = config.DATASET_PATH, 
                            early_stop = config.early_stop,
                            model_name = "MGN_NET",
                            n_folds = config.number_of_folds,
                            fed = True,
                            loss_table_list = loss_list_fed10,
                            loss_vs_epoch = loss_compare_list,
                            rep_vs_epoch = rep_loss_compare_list,
                            kl_vs_epoch = kl_loss_compare_list,
                            update_freq = 10)

    print("********* With federation *********")
    print("update frequency = 1000")
    torch.manual_seed(35813)
    np.random.seed(35813)
    random.seed(35813)
    loss_list_fed1000 = []
    MGN_NET.train_model(n_max_epochs = config.n_max_epochs, 
                            data_path = config.DATASET_PATH, 
                            early_stop = config.early_stop,
                            model_name = "MGN_NET",
                            n_folds = config.number_of_folds,
                            fed = True,
                            loss_table_list = loss_list_fed1000,
                            loss_vs_epoch = loss_compare_list,
                            rep_vs_epoch = rep_loss_compare_list,
                            kl_vs_epoch = kl_loss_compare_list,
                            update_freq = 1000)

    np.save("./output/loss_vs_epoch", loss_compare_list)
    np.save("./output/rep_loss_vs_epoch", rep_loss_compare_list)
    np.save("./output/kl_loss_vs_epoch", kl_loss_compare_list)
    np.save("./output/loss_list_non_fed", loss_list_non_fed)
    np.save("./output/loss_list_fed1", loss_list_fed1)
    np.save("./output/loss_list_fed10", loss_list_fed10)
    np.save("./output/loss_list_fed1000", loss_list_fed1000)

    helper.plotLosses(loss_list_non_fed, loss_list_fed1, loss_list_fed10, loss_list_fed1000)
    helper.plotLossesCompare(loss_compare_list, 0)
    helper.plotLossesCompare(rep_loss_compare_list, 1)
    helper.plotLossesCompare(kl_loss_compare_list, 2)

    #Replot loss
    loss_list_non_fed = np.load("./output/loss_list_non_fed.npy")
    loss_list_fed1 = np.load("./output/loss_list_fed1.npy")
    loss_list_fed10 = np.load("./output/loss_list_fed10.npy")
    loss_list_fed1000 = np.load("./output/loss_list_fed1000.npy")
    helper.plotLosses(loss_list_non_fed, loss_list_fed1, loss_list_fed10, loss_list_fed1000)

    #Replot loss_vs_epoch
    loss_compare_list = np.load("./output/loss_vs_epoch.npy")
    rep_loss_compare_list = np.load("./output/rep_loss_vs_epoch.npy")
    kl_loss_compare_list = np.load("./output/kl_loss_vs_epoch.npy")
    helper.plotLossesCompare(loss_compare_list, 0)
    helper.plotLossesCompare(rep_loss_compare_list, 1)
    helper.plotLossesCompare(kl_loss_compare_list, 2)



# Final results graphing
# def addlabels(x,y, lower):
#     for i in range(len(x)):
#         plt.text(i, (lower + y[i])/2 ,round(y[i], 3), ha = 'center', zorder = 20)

# final_result = [("IID-ASD-LH", [11.550743182500204, 11.08506178855896, 11.0829017162323, 11.074352264404297, 11.058459043502808]), ("Non-IID-ASD-LH", [11.4338, 11.206498384475708, 11.084848642349243, 11.083147287368774, 11.08228611946106]), ("IID-NC-LH", [11.5625, 11.120790958404541, 11.118134021759033, 11.087388277053833, 11.08730173110962]), ("Non-IID-NC-LH", [11.2603, 11.174057245254517, 11.096062183380127, 11.095869779586792, 11.088101387023926])]

# def plotlist_without_non_fed(name_list_pair, include_non_fed):
#     (name, list) = name_list_pair
#     if not include_non_fed:
#         list = list[1:]
#     model_list = ["NonFed", "FedAvg", "FedTW", "FedRank", "FedRank'"]
#     color_list = ['khaki', 'pink', 'skyblue', 'lawngreen', 'salmon']
    
#     diff = max(list) - min(list)
#     lower = min(list) - 0.1*diff
#     upper = max(list) + 0.1*diff
#     plt.ylim(lower, upper)
#     plt.ylabel("FROBENIUS DISTANCE")
#     plt.title(name)
#     plt.grid(axis="y", zorder = -1)
#     # Including non-fed
#     if include_non_fed:
#         plt.bar(model_list, list, color = color_list, zorder = 3)
#         addlabels(model_list, list, lower)
#     # Excluding non-fed
#     else:
#         plt.bar(model_list[1:], list, color = color_list[1:], zorder = 3)
#         addlabels(model_list[1:], list, lower)
#     plt.show()
    
# plotlist_without_non_fed(final_result[3], True)