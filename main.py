from model import MGN_NET
import config
import scipy
import helper
import torch
import random
import numpy as np

# Load dataset from MAT files
helper.clear_dir("input")
# helper.load_input_from_dir_of_mats("data_nc_asd_L/NC LH")
helper.load_input_from_dir_of_mats("data_nc_asd_L/ASD LH")

loss_compare_list = helper.loss_compare_list_init(config.number_of_folds, config.number_of_clients, config.n_max_epochs)
rep_loss_compare_list = helper.loss_compare_list_init(config.number_of_folds, config.number_of_clients, config.n_max_epochs)
kl_loss_compare_list = helper.loss_compare_list_init(config.number_of_folds, config.number_of_clients, config.n_max_epochs)

helper.clear_output()

if config.test_func:
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
    np.save("./output/loss_list_fed1", loss_list_fed1)
    loss_list_fed1 = np.load("./output/loss_list_fed1.npy")
    helper.plotSingleLoss(loss_list_fed1)
    
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

