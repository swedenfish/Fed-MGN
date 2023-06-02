import torch
import helper
import config
from config import number_of_clients, early_stop_rounds, temporal_weighting, average_all
import copy
import random
import uuid
import os
import shutil
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import NNConv
import time
import sys
from torch.nn import Sequential, Linear, ReLU
from sklearn.cluster import KMeans

#set seed for reproducibility
torch.manual_seed(35813)
np.random.seed(35813)
random.seed(35813)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



#check if any gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MGN_NET(torch.nn.Module):
    def __init__(self, dataset):
        super(MGN_NET, self).__init__()
        
        model_params = config.PARAMS
        
        # nn is the non linear neural network layer after message passing
        # so each node with 6 attributes -> 36 attributes as representation
        nn = Sequential(Linear(model_params["Linear1"]["in"], model_params["Linear1"]["out"]), ReLU())
        self.conv1 = NNConv(model_params["conv1"]["in"], model_params["conv1"]["out"], nn, aggr='mean')
        
        nn = Sequential(Linear(model_params["Linear2"]["in"], model_params["Linear2"]["out"]), ReLU())
        self.conv2 = NNConv(model_params["conv2"]["in"], model_params["conv2"]["out"], nn, aggr='mean')
        
        nn = Sequential(Linear(model_params["Linear3"]["in"], model_params["Linear3"]["out"]), ReLU())
        self.conv3 = NNConv(model_params["conv3"]["in"], model_params["conv3"]["out"], nn, aggr='mean')
        
        # nn = Sequential(Linear(6, 36), ReLU())
        # self.conv1 = NNConv(1, 36, nn, aggr='mean')
        
        # nn = Sequential(Linear(6, 36*24), ReLU())
        # self.conv2 = NNConv(36, 24, nn, aggr='mean')
        
        # nn = Sequential(Linear(6, 24*8), ReLU())
        # self.conv3 = NNConv(24, 8, nn, aggr='mean')
        
        # Size of each sample
        # 1 -> 36 -> 24 -> 8
        
    def forward(self, data):
        """
            Args:
                data (Object): data object consist of three parts x, edge_attr, and edge_index.
                                This object can be produced by using helper.cast_data function
                        x: Node features with shape [number_of_nodes, 1] (Simply set to vector of ones since we dont have any)
                        edge_attr: Edge features with shape [number_of_edges, number_of_views]
                        edge_index: Graph connectivities with shape [2, number_of_edges] (COO format) 
                        
        """
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        
        repeated_out = x.repeat(35,1,1)
        repeated_t   =  torch.transpose(repeated_out, 0, 1)
        diff = torch.abs(repeated_out - repeated_t)
        cbt = torch.sum(diff, 2)
        
        return cbt
    
    def prepare_client_dicts(data_path, n_folds, i, weighted_loss):
        """
            Prepare all dicts used for multiple clients to train their models
            Args:
                data_path (string): file path for the dataset 
                n_folds (int): number of cross validation folds
                i (int): current fold index
                weighted_loss (bool): view normalization in centeredness loss
            Return:
                models: trained models 
        """
        n_attr = config.Nattr
        dataset = "simulated"
        model_params = config.PARAMS
        if(n_folds > 1):
            all_train_data, test_data, _, _ = helper.preprocess_data_array(data_path,
                                number_of_folds=n_folds, current_fold_id=i)
            # print(np.shape(all_train_data))
            # print(np.shape(test_data))
        elif(n_folds == 1):
            all_data = np.load(data_path)
            np.random.shuffle(all_data)
            length = all_data.shape[0]
            all_train_data = all_data[:round(length*0.8)]
            test_data = all_data[round(length*0.8):]
        else:
            print("n_folds error: " + n_folds)
        
        model_dict = []
        train_data_dict = []
        train_casted_dict = []
        targets_dict = []    
        loss_weightes_dict = []
        optimizer_dict = []
        test_errors_rep_dict = []
        
        validation_data = all_train_data[0:10]
        all_train_data = all_train_data[10:]
        
        number_of_data = all_train_data.shape[0]
        
        for n in range(number_of_clients):
            
            model = MGN_NET(dataset)
            model = model.to(device)
            model_dict.append(model)
            
            size = number_of_data // number_of_clients 
            if(n==number_of_clients-1):
                train_data = all_train_data[n*size:]
            else:
                train_data = all_train_data[n*size : (n+1)*size]
            train_data_dict.append(train_data)
            
            train_casted = [d.to(device) for d in helper.cast_data(train_data)]
            train_casted_dict.append(train_casted)
            
            # each client's targets are just all its train data
            # will randomly choose subset of it to evaluate loss
            targets =  [torch.tensor(tensor, dtype = torch.float32).to(device) for tensor in train_data]
            targets_dict.append(targets)

            train_mean = np.mean(train_data, axis=(0,1,2))
            #whether rep loss should be calculated with normalised views weight
            if weighted_loss:
                loss_weightes = torch.tensor(np.array(list((1 / train_mean) / np.max(1 / train_mean))*len(train_data)), dtype = torch.float32)
            else:
                loss_weightes =  torch.tensor(np.ones((n_attr*len(train_data))), dtype = torch.float32)
            loss_weightes = loss_weightes.to(device)
            loss_weightes_dict.append(loss_weightes)

            optimizer = torch.optim.AdamW(model.parameters(), lr=model_params["learning_rate"], weight_decay= 0.00)
            optimizer_dict.append(optimizer)

            test_errors_rep = []
            test_errors_rep_dict.append(test_errors_rep)
        test_casted = [d.to(device) for d in helper.cast_data(test_data)]
        validation_casted = [d.to(device) for d in helper.cast_data(validation_data)]
        return (model_dict, train_data_dict, train_casted_dict, targets_dict, loss_weightes_dict, optimizer_dict, test_errors_rep_dict, test_casted, validation_casted)
    
    
    # Not used
    def prepare_client_dicts_using_clusters(data_path, n_folds, i, weighted_loss):
        """
            Very similar to prepare_client_dicts, but use a k-means cluster algorithm to decide global test set and each client's train_data
            Prepare all dicts used for multiple clients to train their models
            Args:
                data_path (string): file path for the dataset 
                n_folds (int): number of cross validation folds
                i (int): current fold index
                weighted_loss (bool): view normalization in centeredness loss
            Return:
                models: trained models 
        """
        n_attr = config.Nattr
        dataset = "simulated"
        model_params = config.PARAMS
            
        model_dict = []
        train_data_dict = []
        train_casted_dict = []
        targets_dict = []    
        loss_weightes_dict = []
        optimizer_dict = []
        test_errors_rep_dict = []
        
        if(n_folds == 1):
            all_data = np.load(data_path)
            length = all_data.shape[0]
            train_data_dict.append(all_data[:round(length*0.8)])
            test_data = all_data[round(length*0.8):]
        elif(n_folds > 1):
            #TODO
            (train_data_dict, test_data) = MGN_NET.k_mean_clustering()
        else:
            print("n_folds error: " + n_folds)
                
                
        # train_data_dict and test_data should be ready till here        
        for n in range(number_of_clients):
            
            model = MGN_NET(dataset)
            model = model.to(device)
            model_dict.append(model)
            
            train_data = train_data_dict[n]
            train_casted = [d.to(device) for d in helper.cast_data(train_data)]
            train_casted_dict.append(train_casted)
            
            # each client's targets are just all its train data
            # will randomly choose subset of it to evaluate loss
            targets =  [torch.tensor(tensor, dtype = torch.float32).to(device) for tensor in train_data]
            targets_dict.append(targets)

            train_mean = np.mean(train_data, axis=(0,1,2))
            if weighted_loss:
                loss_weightes = torch.tensor(np.array(list((1 / train_mean) / np.max(1 / train_mean))*len(train_data)), dtype = torch.float32)
            else:
                loss_weightes =  torch.tensor(np.ones((n_attr*len(train_data))), dtype = torch.float32)
            loss_weightes = loss_weightes.to(device)
            loss_weightes_dict.append(loss_weightes)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=model_params["learning_rate"], weight_decay= 0.00)
            optimizer_dict.append(optimizer)
            
            test_errors_rep = []
            test_errors_rep_dict.append(test_errors_rep)
        test_casted = [d.to(device) for d in helper.cast_data(test_data)]
        return (model_dict, train_data_dict, train_casted_dict, targets_dict, loss_weightes_dict, optimizer_dict, test_errors_rep_dict, test_casted)
    
    #TODO Not used
    def k_mean_clustering():
        return 
    
    
    def simulate_dataset_using_clustering(data_path, number_of_folds, number_of_clients):
        """
        This function acts as a preprocessing function before doing training
        Accepts a dataset.npy representing all data, and split it to k folds.
        Each fold will then be split into n clusters using k-means, which simulates data in n clients
        20% of each clusters will be taken out and mix together as a global testset
        """
        all_data = np.load(data_path)
      
    def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, \
                                                   number_of_samples, clients_with_access):
        '''
        This function updates clients with the newly global model so that all the devices can take advantage of the common
        information.
        '''
        with torch.no_grad():
            for i in range(number_of_samples): # clients_with_access 

                model_dict[i].conv1.nn[0].weight.data = 0.5*(main_model.conv1.nn[0].weight.data.clone() + model_dict[i].conv1.nn[0].weight.data.clone())
                model_dict[i].conv1.nn[0].bias.data = 0.5*(main_model.conv1.nn[0].bias.data.clone() + model_dict[i].conv1.nn[0].bias.data.clone())
                model_dict[i].conv1.bias.data = 0.5*(main_model.conv1.bias.data.clone() + model_dict[i].conv1.bias.data.clone())
                model_dict[i].conv1.lin.weight.data = 0.5*(main_model.conv1.lin.weight.data.clone() + model_dict[i].conv1.lin.weight.data.clone())
                
                model_dict[i].conv2.nn[0].weight.data = 0.5*(main_model.conv2.nn[0].weight.data.clone() + model_dict[i].conv2.nn[0].weight.data.clone())
                model_dict[i].conv2.nn[0].bias.data = 0.5*(main_model.conv2.nn[0].bias.data.clone() + model_dict[i].conv2.nn[0].bias.data.clone())
                model_dict[i].conv2.bias.data = 0.5*(main_model.conv2.bias.data.clone() + model_dict[i].conv2.bias.data.clone())
                model_dict[i].conv2.lin.weight.data = 0.5*(main_model.conv2.lin.weight.data.clone() + model_dict[i].conv2.lin.weight.data.clone())
                
                model_dict[i].conv3.nn[0].weight.data = 0.5*(main_model.conv3.nn[0].weight.data.clone() + model_dict[i].conv3.nn[0].weight.data.clone())
                model_dict[i].conv3.nn[0].bias.data = 0.5*(main_model.conv3.nn[0].bias.data.clone() + model_dict[i].conv3.nn[0].bias.data.clone())
                model_dict[i].conv3.bias.data = 0.5*(main_model.conv3.bias.data.clone() + model_dict[i].conv3.bias.data.clone())
                model_dict[i].conv3.lin.weight.data = 0.5*(main_model.conv3.lin.weight.data.clone() + model_dict[i].conv3.lin.weight.data.clone())
                
        return model_dict
    
    def cal_weight_diff(main_model, model):
        """
        This function calculates the weight difference between global model and client mode in order to calculate the proximal loss
        """
        client_model = model
        
        w1 = client_model.conv1.nn[0].weight
        w2 = client_model.conv2.nn[0].weight
        w3 = client_model.conv3.nn[0].weight
        w4 = client_model.conv1.lin.weight
        w5 = client_model.conv2.lin.weight
        w6 = client_model.conv3.lin.weight
        
        main_model_w1 = main_model.conv1.nn[0].weight.data.clone().detach()
        main_model_w2 = main_model.conv2.nn[0].weight.data.clone().detach()
        main_model_w3 = main_model.conv3.nn[0].weight.data.clone().detach()
        main_model_w4 = main_model.conv1.lin.weight.data.clone().detach()
        main_model_w5 = main_model.conv2.lin.weight.data.clone().detach()
        main_model_w6 = main_model.conv3.lin.weight.data.clone().detach()
        
        # torch.Size([36, 6])
        # torch.Size([864, 6])
        # torch.Size([192, 6])
        # torch.Size([36, 1])
        # torch.Size([24, 36])
        # torch.Size([8, 24])
        # torch.Size([36, 6])
        # torch.Size([864, 6])
        # torch.Size([192, 6])
        # torch.Size([36, 1])
        # torch.Size([24, 36])
        # torch.Size([8, 24])

        # print(w1.shape)
        # print(w2.shape)
        # print(w3.shape)
        # print(w4.shape)
        # print(w5.shape)
        # print(w6.shape)
        # print(main_model_w1.shape)
        # print(main_model_w2.shape)
        # print(main_model_w3.shape)
        # print(main_model_w4.shape)
        # print(main_model_w5.shape)
        # print(main_model_w6.shape)
        
        diff_sum = torch.sum(torch.abs(torch.subtract(w1, main_model_w1)))
        diff_sum = torch.add(diff_sum, torch.sum(torch.abs(torch.subtract(w2, main_model_w2))))
        diff_sum = torch.add(diff_sum, torch.sum(torch.abs(torch.subtract(w3, main_model_w3))))
        diff_sum = torch.add(diff_sum, torch.sum(torch.abs(torch.subtract(w4, main_model_w4))))
        diff_sum = torch.add(diff_sum, torch.sum(torch.abs(torch.subtract(w5, main_model_w5))))
        diff_sum = torch.add(diff_sum, torch.sum(torch.abs(torch.subtract(w6, main_model_w6))))

        return diff_sum
        
        
        
    def set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, \
                                                                     number_of_samples, clients_with_access, \
                                                                    last_updated_dict, current_epoch):
        '''
        This function takes combined weights for global model and assigns them to the global model.
        '''
        conv1_nn_mean_weight, conv1_nn_mean_bias, conv1_bias, conv1_root_weight, conv1_root_bias, \
        conv2_nn_mean_weight, conv2_nn_mean_bias, conv2_bias, conv2_root_weight, conv2_root_bias, \
        conv3_nn_mean_weight, conv3_nn_mean_bias, conv3_bias, conv3_root_weight, conv3_root_bias = MGN_NET.get_averaged_weights(model_dict, \
                                                        number_of_samples, clients_with_access, \
                                                        average_all, last_updated_dict, current_epoch)
        
        with torch.no_grad():
            main_model.conv1.nn[0].weight.data = conv1_nn_mean_weight.clone()
            main_model.conv1.nn[0].bias.data = conv1_nn_mean_bias.clone()
            main_model.conv1.bias.data = conv1_bias.clone()
            main_model.conv1.lin.weight.data = conv1_root_weight.clone()
            
            main_model.conv2.nn[0].weight.data = conv2_nn_mean_weight.clone()
            main_model.conv2.nn[0].bias.data = conv2_nn_mean_bias.clone()
            main_model.conv2.bias.data = conv2_bias.clone()
            main_model.conv2.lin.weight.data = conv2_root_weight.clone()
            
            main_model.conv3.nn[0].weight.data = conv3_nn_mean_weight.clone()
            main_model.conv3.nn[0].bias.data = conv3_nn_mean_bias.clone()
            main_model.conv3.bias.data = conv3_bias.clone()
            main_model.conv3.lin.weight.data = conv3_root_weight.clone()

        return main_model

    def get_averaged_weights(model_dict, number_of_samples, clients_with_access, \
                         average_all=True, last_updated_dict=None, current_epoch=-1):
        '''
        This function averages model weights after a designated number of round so that we can have the weights of the global model
        that takes full advantage of introduced devices in the federated pipeline.
        '''
        size_1 = list(model_dict[0].conv1.lin.weight.shape)
        size_2 = list(model_dict[0].conv2.lin.weight.shape)
        size_3 = list(model_dict[0].conv3.lin.weight.shape)
        
        conv1_nn_mean_weight = torch.zeros(size=model_dict[0].conv1.nn[0].weight.data.shape).to(device)
        conv1_nn_mean_bias = torch.zeros(size=model_dict[0].conv1.nn[0].bias.data.shape).to(device)
        conv1_bias = torch.zeros(size=model_dict[0].conv1.bias.data.shape).to(device)
        conv1_root_weight = torch.zeros(size=model_dict[0].conv1.lin.weight.data.shape).to(device)
        conv1_root_bias = 0
        
        conv2_nn_mean_weight = torch.zeros(size=model_dict[0].conv2.nn[0].weight.data.shape).to(device)
        conv2_nn_mean_bias = torch.zeros(size=model_dict[0].conv2.nn[0].bias.data.shape).to(device)
        conv2_bias = torch.zeros(size=model_dict[0].conv2.bias.data.shape).to(device)
        conv2_root_weight = torch.zeros(size=model_dict[0].conv2.lin.weight.data.shape).to(device)
        conv2_root_bias = 0
        
        conv3_nn_mean_weight = torch.zeros(size=model_dict[0].conv3.nn[0].weight.data.shape).to(device)
        conv3_nn_mean_bias = torch.zeros(size=model_dict[0].conv3.nn[0].bias.data.shape).to(device)
        conv3_bias = torch.zeros(size=model_dict[0].conv3.bias.data.shape).to(device)
        conv3_root_weight = torch.zeros(size=model_dict[0].conv3.lin.weight.data.shape).to(device)
        conv3_root_bias = 0
        
        cls = None # 
        if average_all:
            cls = range(number_of_samples)
        else:
            cls = clients_with_access

        with torch.no_grad():
            def getWeight_i(i):
                return ((np.e / 2) ** (- (current_epoch - last_updated_dict['client'+str(i)])))
            for i in cls: # cls
                if temporal_weighting == True:
                    all_weights = getWeight_i(0) + getWeight_i(1) + getWeight_i(2)
                    client_weight = getWeight_i(i) / all_weights
                else:
                    client_weight = 1/len(cls)

                conv1_nn_mean_weight += (client_weight * model_dict[i].conv1.nn[0].weight.data.clone())
                conv1_nn_mean_bias += (client_weight * model_dict[i].conv1.nn[0].bias.data.clone())
                conv1_bias += (client_weight * model_dict[i].conv1.bias.data.clone())
                conv1_root_weight += (client_weight * model_dict[i].conv1.lin.weight.data.clone())
                
                conv2_nn_mean_weight += (client_weight * model_dict[i].conv2.nn[0].weight.data.clone())
                conv2_nn_mean_bias += (client_weight * model_dict[i].conv2.nn[0].bias.data.clone())
                conv2_bias += (client_weight * model_dict[i].conv2.bias.data.clone())
                conv2_root_weight += (client_weight * model_dict[i].conv2.lin.weight.data.clone())
                
                conv3_nn_mean_weight += (client_weight * model_dict[i].conv3.nn[0].weight.data.clone())
                conv3_nn_mean_bias += (client_weight * model_dict[i].conv3.nn[0].bias.data.clone())
                conv3_bias += (client_weight * model_dict[i].conv3.bias.data.clone())
                conv3_root_weight += (client_weight * model_dict[i].conv3.lin.weight.data.clone())

        return conv1_nn_mean_weight, conv1_nn_mean_bias, conv1_bias, conv1_root_weight, conv1_root_bias, \
                conv2_nn_mean_weight, conv2_nn_mean_bias, conv2_bias, conv2_root_weight, conv2_root_bias, \
                conv3_nn_mean_weight, conv3_nn_mean_bias, conv3_bias, conv3_root_weight, conv3_root_bias
    
    @staticmethod
    def generate_subject_biased_cbts(model, train_data):
        """
            Generates all possible CBTs for a given training set.
            Args:
                model: trained DGN model
                train_data: list of data objects
        """
        model.eval()
        cbts = np.zeros((35,35, len(train_data)))
        train_data = [d.to(device) for d in train_data]
        for i, data in enumerate(train_data):
            cbt = model(data)
            cbts[:,:,i] = np.array(cbt.cpu().detach())
        
        return cbts
        
    @staticmethod
    def generate_cbt_median(model, train_data):
        """
            Generate optimized CBT for the training set (use post training refinement)
            Args:
                model: trained DGN model
                train_data: list of data objects
        """
        model.eval()
        cbts = []
        train_data = [d.to(device) for d in train_data]
        for data in train_data:
            cbt = model(data)
            cbts.append(np.array(cbt.cpu().detach()))
        final_cbt = torch.tensor(np.median(cbts, axis = 0), dtype = torch.float32).to(device)
        
        return final_cbt  
    
    @staticmethod
    def KL_error(cbt, target_data, six_views = False):
        """
            Calculate the KL_divergence between the CBT and test subjects (all views)
            Args:
                cbt: models output
                target_data: list of data objects
        """
        cbt_dist = cbt.sum(axis = 1)
        cbt_probs = cbt_dist / cbt_dist.sum()
        
        views = torch.cat([data.con_mat for data in target_data], axis = 2).permute((2,1,0))
        #View 1
        view1_mean = views[range(0,views.shape[0],6 if six_views else 4)].mean(axis = 0)
        view1_dist = view1_mean.sum(axis = 1)
        view1_prob = view1_dist / view1_dist.sum()
        kl_1 = ((cbt_probs * torch.log2(cbt_probs/view1_prob)).sum().abs()) +  ((view1_prob * torch.log2(view1_prob/cbt_probs)).sum().abs())
        
        #View 2
        view2_mean = views[range(1,views.shape[0],6 if six_views else 4)].mean(axis = 0)
        view2_dist = view2_mean.sum(axis = 1)
        view2_prob = view2_dist / view2_dist.sum()
        kl_2 = ((cbt_probs * torch.log2(cbt_probs/view2_prob)).sum().abs()) +  ((view2_prob * torch.log2(view2_prob/cbt_probs)).sum().abs())
        
        #View 3
        view3_mean = views[range(2,views.shape[0],6 if six_views else 4)].mean(axis = 0)
        view3_dist = view3_mean.sum(axis = 1)
        view3_prob = view3_dist / view3_dist.sum()
        kl_3 = ((cbt_probs * torch.log2(cbt_probs/view3_prob)).sum().abs()) +  ((view3_prob * torch.log2(view3_prob/cbt_probs)).sum().abs())
        
        #View 4
        view4_mean = views[range(3,views.shape[0],6 if six_views else 4)].mean(axis = 0) 
        view4_dist = view4_mean.sum(axis = 1)
        view4_prob = view4_dist / view4_dist.sum()
        kl_4 = ((cbt_probs * torch.log2(cbt_probs/view4_prob)).sum().abs()) + ((view4_prob * torch.log2(view4_prob/cbt_probs)).sum().abs())
        
        if six_views:
            #View 5
            view5_mean = views[range(4,views.shape[0],6 if six_views else 4)].mean(axis = 0)
            view5_dist = view5_mean.sum(axis = 1)
            view5_prob = view5_dist / view5_dist.sum()
            kl_5 = ((cbt_probs * torch.log2(cbt_probs/view5_prob)).sum().abs()) +  ((view5_prob * torch.log2(view5_prob/cbt_probs)).sum().abs())
            
            #View 6
            view6_mean = views[range(5,views.shape[0],6 if six_views else 4)].mean(axis = 0) 
            view6_dist = view6_mean.sum(axis = 1)
            view6_prob = view6_dist / view6_dist.sum()
            kl_6 = ((cbt_probs * torch.log2(cbt_probs/view6_prob)).sum().abs()) + ((view6_prob * torch.log2(view6_prob/cbt_probs)).sum().abs())
        else:
            kl_5, kl_6 = 0, 0 
        return kl_1, kl_2, kl_3, kl_4, kl_5, kl_6
        
    @staticmethod
    def mean_frobenious_distance(generated_cbt, test_data):
        """
            Calculate the mean Frobenious distance between the CBT and test subjects (all views)
            Args:
                generated_cbt: trained DGN model
                test_data: list of data objects
        """
        frobenius_all = []
        # print(len(test_data)) 47 so didn't randomly choose?
        for data in test_data:
            views = data.con_mat
            # views -> [35,35,6] simply the subject?
            for index in range(views.shape[2]):
                diff = torch.abs(views[:,:,index] - generated_cbt)
                diff = diff*diff
                sum_of_all = diff.sum()
                d = torch.sqrt(sum_of_all)
                frobenius_all.append(d)
        return sum(frobenius_all) / len(frobenius_all)
    
    @staticmethod
    def train_model(n_max_epochs, data_path, early_stop, model_name, fed, loss_table_list, loss_vs_epoch, rep_vs_epoch, kl_vs_epoch, weighted_loss = config.weighted_loss, random_sample_size_para = config.N_RANDOM_SAMPLES, n_folds = 5, update_freq = 1):
        """
            Trains a model for each cross validation fold and 
            saves all models along with CBTs to ./output/<model_name> 
            Args:
                n_max_epochs (int): number of training epochs (if early_stop == True this is maximum epoch limit)
                data_path (string): file path for the dataset 
                early_stop (bool): if set true, model will stop training when overfitting starts.
                model_name (string): name for saving the model
                weighted (bool): view normalization in centeredness loss
                random_sample_size (int): random subset size for SNL function
                n_folds (int): number of cross validation folds
            Return:
                models: trained models 
        """
        
        models = []
        n_attr = config.Nattr
        dataset = "simulated"
        
        save_path = config.MODEL_WEIGHT_BACKUP_PATH + "/" + model_name + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.isdir("temp"):
            os.makedirs("temp")
        if not os.path.exists('output/' + "CBT_images"):
            os.mkdir('output/' + "CBT_images")
        helper.clear_dir("temp")
        helper.clear_dir(save_path)
        
            
        model_id = str(uuid.uuid4())
        model_params = config.PARAMS
        # Same shape with loss_compare_list
        validation_error = helper.loss_compare_list_init(config.number_of_folds, config.number_of_clients, config.n_max_epochs)
        
        for i in range(n_folds):
            print("********* FOLD {} *********".format(i))
            # Store each clients' final loss in this fold
            loss_table = []
            
            # Main model in the sever
            main_model = MGN_NET(dataset)
            main_model = main_model.to(device)
            main_optimizer = torch.optim.AdamW(main_model.parameters(), lr=model_params["learning_rate"], weight_decay= 0.00)
            
            model_dict, _, train_casted_dict, targets_dict, loss_weightes_dict, optimizer_dict, test_errors_rep_dict, test_casted, validation_casted = MGN_NET.prepare_client_dicts(data_path, n_folds, i, weighted_loss)
            # 38 38 40 for 3 clients in ASD  39 is test (no validation)
            # 155/4 -> 39 as test set, 116 as training. 10 as validation -> 106 for training. -> 35, 35, 36 
            # print(str(len(train_casted_dict[0])) + " train_data")
            # print(str(len(train_casted_dict[1])) + " train_data")
            # print(str(len(train_casted_dict[2])) + " train_data")
            number_views = config.number_of_views
            tick = time.time()
            early_stop_dict = [False] * number_of_clients
            
            #Ready to start
            for epoch in range(n_max_epochs):
                
                if fed and epoch % update_freq == 0:
                    # send models to all clients
                    MGN_NET.send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, \
                                                                number_of_clients, list(range(0, number_of_clients)))
                    
                #     # print("main model updated, and send to all clients")
                    
                # Each client trains its model locally
                for j in [i for i, x in enumerate(early_stop_dict) if not x]:
                    model = model_dict[j]
                    model.train()
                    losses = []
                    rep_losses = []
                    kl_losses = []
                    train_casted = train_casted_dict[j]
                    targets = targets_dict[j]
                    random_sample_size = int(random_sample_size_para/number_of_clients)
                    loss_weightes = loss_weightes_dict[j]
                    optimizer = optimizer_dict[j]
                    
                    for data in train_casted:
                        cbt = model(data)
                        views_sampled = random.sample(targets, random_sample_size)
                        sampled_targets = torch.cat(views_sampled, axis = 2).permute((2,1,0))
                        expanded_cbt = cbt.expand((sampled_targets.shape[0],35,35))
                        diff = torch.abs(expanded_cbt - sampled_targets) #Absolute difference
                        sum_of_all = torch.mul(diff, diff).sum(axis = (1,2)) #Sum of squares
                        l = torch.sqrt(sum_of_all)  #Square root of the sum
                        cbt_dist = cbt.sum(axis = 1)
                        cbt_probs = cbt_dist / cbt_dist.sum()
                        
                        #View 1 target
                        target_mean1 = sampled_targets[range(0,random_sample_size * number_views, number_views)].mean(axis = 0)
                        target_dist1 = target_mean1.sum(axis = 1)
                        target_probs1 = target_dist1 / target_dist1.sum()
                        kl_loss_1 = ((cbt_probs * torch.log2(cbt_probs/target_probs1)).sum().abs()) + ((target_probs1* torch.log2(target_probs1/cbt_probs)).sum().abs())
                        
                        #View 2 target
                        target_mean2 = sampled_targets[range(1,random_sample_size * number_views, number_views)].mean(axis = 0)
                        target_dist2 = target_mean2.sum(axis = 1)
                        target_probs2 = target_dist2 / target_dist2.sum()
                        kl_loss_2 = ((cbt_probs * torch.log2(cbt_probs/target_probs2)).sum().abs()) + ((target_probs2 * torch.log2(target_probs2/cbt_probs)).sum().abs())
                        
                        #View 3 target
                        target_mean3 = sampled_targets[range(2,random_sample_size * number_views, number_views)].mean(axis = 0)
                        target_dist3 = target_mean3.sum(axis = 1)
                        target_probs3 = target_dist3 / target_dist3.sum()
                        kl_loss_3 = ((cbt_probs * torch.log2(cbt_probs/target_probs3)).sum().abs()) + ((target_probs3* torch.log2(target_probs3/cbt_probs)).sum().abs())
                        
                        #View 4 target
                        target_mean4 = sampled_targets[range(3,random_sample_size * number_views, number_views)].mean(axis = 0)
                        target_dist4 = target_mean4.sum(axis = 1)
                        target_probs4 = target_dist4 / target_dist4.sum()
                        kl_loss_4 = ((cbt_probs * torch.log2(cbt_probs/target_probs4)).sum().abs()) + ((target_probs4* torch.log2(target_probs4/cbt_probs)).sum().abs())
                        
                        if number_views == 6:
                            #View 5 target
                            target_mean5 = sampled_targets[range(4,random_sample_size * number_views, number_views)].mean(axis = 0)
                            target_dist5 = target_mean5.sum(axis = 1)
                            target_probs5 = target_dist5 / target_dist5.sum()
                            kl_loss_5 = ((cbt_probs * torch.log2(cbt_probs/target_probs5)).sum().abs()) + ((target_probs5* torch.log2(target_probs5/cbt_probs)).sum().abs())
                            
                            #View 6 target
                            target_mean6 = sampled_targets[range(5,random_sample_size * number_views, number_views)].mean(axis = 0)
                            target_dist6 = target_mean6.sum(axis = 1)
                            target_probs6 = target_dist6 / target_dist6.sum()
                            kl_loss_6 = ((cbt_probs * torch.log2(cbt_probs/target_probs6)).sum().abs()) + ((target_probs6* torch.log2(target_probs6/cbt_probs)).sum().abs())
                        else:
                            kl_loss_5 = 0
                            kl_loss_6 = 0
                        kl_loss = (kl_loss_1 + kl_loss_2 + kl_loss_3 + kl_loss_4 + kl_loss_5 + kl_loss_6)
                        # use weighting to avoid bias towards high value views
                        rep_loss = (l * loss_weightes[:random_sample_size * n_attr]).mean()
                        kl_losses.append(kl_loss)
                        rep_losses.append(rep_loss)
                        losses.append(kl_loss * model_params["lambda_kl"] + rep_loss)
                        
                    loss = torch.mean(torch.stack(losses))
                    if fed and config.prox:
                        weight_diff = MGN_NET.cal_weight_diff(main_model, model)
                        loss = torch.add(loss, torch.mul(torch.mul(weight_diff, weight_diff), config.mu * 0.5))
                    kl_loss = torch.mean(torch.stack(kl_losses))
                    rep_loss = torch.mean(torch.stack(rep_losses))
                    if not fed:
                        loss_vs_epoch[i][j][0][epoch] = loss
                        rep_vs_epoch[i][j][0][epoch] = rep_loss
                        kl_vs_epoch[i][j][0][epoch] = kl_loss
                    elif update_freq==1:
                        loss_vs_epoch[i][j][1][epoch] = loss
                        rep_vs_epoch[i][j][1][epoch] = rep_loss
                        kl_vs_epoch[i][j][1][epoch] = kl_loss
                    elif update_freq==10:
                        loss_vs_epoch[i][j][2][epoch] = loss
                        rep_vs_epoch[i][j][2][epoch] = rep_loss
                        kl_vs_epoch[i][j][2][epoch] = kl_loss
                    elif update_freq==1000:
                        loss_vs_epoch[i][j][3][epoch] = loss
                        rep_vs_epoch[i][j][3][epoch] = rep_loss
                        kl_vs_epoch[i][j][3][epoch] = kl_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                if fed and epoch % update_freq == 0:
                    all_clents = list(range(0, number_of_clients))
                    portion = 0.1
                    non_stragglers = random.sample(all_clents, int(portion * number_of_clients))
                    non_stragglers.sort()
                    # update main models from all parameters received
                    main_model = MGN_NET.set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, \
                                      model_dict, number_of_clients, \
                                        non_stragglers, None, epoch+1)           
                
                    # # send models to all clients
                    # MGN_NET.send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, \
                    #                                             number_of_clients, list(range(0, number_of_clients)))
                    
                    # print("main model updated, and send to all clients")
                for j in [i for i, x in enumerate(early_stop_dict) if not x]:
                    model = model_dict[j]
                    train_casted = train_casted_dict[j]
                    cbt = MGN_NET.generate_cbt_median(model, train_casted)
                    rep_loss = MGN_NET.mean_frobenious_distance(cbt, validation_casted)
                    kl1, kl2, kl3, kl4, kl5, kl6 = MGN_NET.KL_error(cbt, validation_casted, six_views = (number_views==6))
                    rep_loss = float(rep_loss)
                    kl_loss = float(kl1+kl2+kl3+kl4+kl5+kl6)
                    error = kl_loss * model_params["lambda_kl"] + rep_loss
                    
                    if not fed:
                        validation_error[i][j][0][epoch] = error
                    elif update_freq==1:
                        validation_error[i][j][1][epoch] = error
                    elif update_freq==10:
                        validation_error[i][j][2][epoch] = error
                    elif update_freq==1000:
                        validation_error[i][j][3][epoch] = error
                    
                # The evaluation error after the epochth training
                if epoch % 10 == 0:
                    for j in [i for i, x in enumerate(early_stop_dict) if not x]:
                        model = model_dict[j]
                        train_casted = train_casted_dict[j]
                        #TODO
                        #test_casted = test_casted_dict[j]
                        test_errors_rep = test_errors_rep_dict[j]
                        cbt = MGN_NET.generate_cbt_median(model, train_casted)
                        rep_loss = MGN_NET.mean_frobenious_distance(cbt, test_casted)
                        kl1, kl2, kl3, kl4, kl5, kl6 = MGN_NET.KL_error(cbt, test_casted, six_views = (number_views==6))
                        tock = time.time()
                        time_elapsed = tock - tick
                        rep_loss = float(rep_loss)
                        test_errors_rep.append(rep_loss)

                        print("Epoch: {}  |  Client: {}  |  {} Rep: {:.2f}  |  KL: {:.2f} | Time Elapsed: {:.2f}  |".format(epoch, j,
                            data_path.split("/")[-1].split(" ")[0], rep_loss, float(kl1+kl2+kl3+kl4+kl5+kl6), time_elapsed))
                        try:
                            #Early stopping and restoring logic
                            torch.save(model.state_dict(), "./temp/weight_" + model_id + "_" + str(rep_loss)[:5]  + ".model")
                            # if len(test_errors_rep) > early_stop_rounds:
                            #     last_errors = test_errors_rep[-early_stop_rounds:]
                            #     if(early_stop and all(last_errors[i+1] < last_errors[i] for i in range(early_stop_rounds-1))):
                            #         print("Client " + str(j) +" Early Stopping")
                            #         early_stop_dict[j] = True
                            
                            
                    # Check for early stopping:\
                            early_stop_interval = config.early_stop_interval
                            if epoch > early_stop_interval:
                                if not fed:
                                    if abs(validation_error[i][j][0][epoch] - validation_error[i][j][0][epoch-early_stop_interval]) < config.early_stop_distance:
                                        print("Client " + str(j) +" Early Stopping")
                                        early_stop_dict[j] = True
                                        continue
                                elif update_freq==1:
                                    if abs(validation_error[i][j][1][epoch] - validation_error[i][j][1][epoch-early_stop_interval]) < config.early_stop_distance:
                                        print("Client " + str(j) +" Early Stopping")
                                        early_stop_dict[j] = True
                                        continue
                                elif update_freq==10:
                                    if abs(validation_error[i][j][2][epoch] - validation_error[i][j][2][epoch-early_stop_interval]) < config.early_stop_distance:
                                        print("Client " + str(j) +" Early Stopping")
                                        early_stop_dict[j] = True
                                        continue
                                elif update_freq==1000:
                                    if abs(validation_error[i][j][3][epoch] - validation_error[i][j][3][epoch-early_stop_interval]) < config.early_stop_distance:
                                        print("Client " + str(j) +" Early Stopping")
                                        early_stop_dict[j] = True
                                        continue
                                    
                        except:
                            print("ERROR occured")
                            break
                    tick = tock
                
                
                            
            for j in range(number_of_clients):
                model = model_dict[j]
                test_errors_rep = test_errors_rep_dict[j]
                train_casted = train_casted_dict[j]
                #TODO
                # test_casted = test_casted_dict[j]
                restore = "./temp/weight_" + model_id + "_" + str(min(test_errors_rep))[:5] + ".model"
                model.load_state_dict(torch.load(restore))
                #Save each client's final model
                torch.save(model.state_dict(), save_path + "fold" + str(i) + " client" + str(j) + ".model")
                models.append(model)
                
                cbt = MGN_NET.generate_cbt_median(model, train_casted)
                rep_loss = MGN_NET.mean_frobenious_distance(cbt, test_casted)
                kl_loss = float(sum(MGN_NET.KL_error(cbt, test_casted, number_views==6)))
                cbt = cbt.cpu().numpy()
                np.save( save_path + "fold" + str(i) + " client" + str(j) + "_cbt", cbt)
                all_cbts = MGN_NET.generate_subject_biased_cbts(model, train_casted)
                np.save(save_path + "fold" + str(i) + " client" + str(j) + "_all_cbts", all_cbts)
                helper.save_cbt(cbt, i, j, fed)
                
                print("FINAL RESULTS  Client {}  REP: {}  KL: {}".format(j, rep_loss, kl_loss))
                
                loss_table.append((rep_loss, kl_loss))
            loss_table_list.append(loss_table)
        return models
