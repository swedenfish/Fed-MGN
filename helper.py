from sklearn.model_selection import KFold
import torch
from torch_geometric.data import Data
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import config
import shutil
from operator import add, truediv

#set seed for reproducibility
torch.manual_seed(35813)
np.random.seed(35813)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def create_better_simulated(N_Subjects, N_ROIs):
    """
        Simulated dataset distributions are inspired from real measurements
        so this function creates better dataset for demo.
        However, number of views are hardcoded.
    """
    features =  np.triu_indices(N_ROIs)[0].shape[0]
    view1 = np.random.normal(0.1,0.069, (N_Subjects, features))
    view1 = view1.clip(min = 0)
    view1 = np.array([antiVectorize(v, N_ROIs) for v in view1])
    
    view2 = np.random.normal(0.72,0.5, (N_Subjects, features))
    view2 = view2.clip(min = 0)
    view2 = np.array([antiVectorize(v, N_ROIs) for v in view2])
    
    view3 = np.random.normal(0.32,0.20, (N_Subjects, features))
    view3 = view3.clip(min = 0)
    view3 = np.array([antiVectorize(v, N_ROIs) for v in view3])
    
    view4 = np.random.normal(0.03,0.015, (N_Subjects, features))
    view4 = view4.clip(min = 0)
    view4 = np.array([antiVectorize(v, N_ROIs) for v in view4])
    
    return np.stack((view1, view2, view3, view4), axis = 3)

def simulate_dataset(N_Subjects, N_ROIs, N_views):
    """
        Creates random dataset
        Args:
            N_Subjects: number of subjects
            N_ROIs: number of region of interests
            N_views: number of views
        Return:
            dataset: random dataset with shape [N_Subjects, N_ROIs, N_ROIs, N_views]
    """
    features =  np.triu_indices(N_ROIs)[0].shape[0]
    views = []
    for _ in range(N_views):
        view = np.random.uniform(0.1,2, (N_Subjects, features))
        
        view = np.array([antiVectorize(v, N_ROIs) for v in view])
        views.append(view)
    return np.stack(views, axis = 3)



def get_std_and_mean(list_of_tensors):
    tensor = np.array(list_of_tensors)
    tensor_means = np.mean(tensor[:,:,:,:], axis=(0,1,2))
    tensor_std =   np.std(tensor[:,:,:,:], axis=(0,1,2))
    return tensor_std, tensor_means


def plot_graphs(losses_array, labels, name):
    x = range(0, len(losses_array[0]) * 10, 10)
    
    if(len(losses_array) == 1):
        plt.plot(x, losses_array[0])
        plt.savefig(name + ".png")
        plt.close()
        return
    
    for losses in losses_array:
        plt.plot(x, losses)
    plt.legend(labels, loc='upper right')
    plt.savefig(name + ".png")
    plt.close()

def generate_same_folds_for_matlab(data_path, n_folds):
    for i in range(n_folds):
            print("********* FOLD {} *********".format(i))
            train_data, test_data, _, _ = preprocess_data_array(data_path, number_of_folds=n_folds, current_fold_id=i)
            
            #Uncomment these three lines to generate exactly same train and test for netNorm
            mdict = {"train": np.array(train_data).swapaxes(3,0), "test": np.array(test_data).swapaxes(3,0)}
            save_path = "/data_{}_{}.mat".format(data_path.split("/")[-1], i)
            scipy.io.savemat("./netNorm/dataset" + save_path , mdict)

def clear_dir(dir_name):
    for file in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, file))

def antiVectorize(vec, m):
    M = np.zeros((m,m))
    t = 0
    for i  in range(0,m - 1):
        for j in range(i+1, m):
            M[i,j] = vec[t]
            M[j,i] = vec[t]
            t = t + 1
    return M

def Vectorize(matrix):
    return matrix[np.triu_indices(matrix.shape[0], k = 1)]

def binary_correspondence(arr1, arr2):
    count = 0
    for a in arr1:
        if (a in arr2):
            count += 1
    return count

def read_all_dataset(root, read_indices = None, connection_mask = None):
    print("reading " + root)
    files = os.listdir(root)
    all_data = []
    #try:
    files = sorted(files, key=lambda f: int(f.split(".mat")[0].split("Sub")[1]))[:155]
    for i, file in enumerate(files):
        if read_indices == None:
            mvbn = scipy.io.loadmat(root + "/" + file)["views"]
            if connection_mask is not None:
                mvbn[connection_mask != 1] = 0
            all_data.append(mvbn)
        else:
            if (i in read_indices):
                mvbn = scipy.io.loadmat(root + "/" + file)["views"]
                if connection_mask is not None:
                    mvbn[connection_mask != 1] = 0
                all_data.append(mvbn)
    
    return [np.array(data) for data in all_data]

def preprocess_data_array(data_path, number_of_folds, current_fold_id):
    X = np.load(data_path)
    # 155 for ASD
    # print(np.shape(X))
    np.random.seed(35813)
    np.random.shuffle(X)
    kf = KFold(n_splits=number_of_folds)
    split_indices = kf.split(range(X.shape[0]))
    train_indices, test_indices = [(list(train), list(test)) for train, test in split_indices][current_fold_id]
    #Split train and test
    X_train = X[train_indices]
    X_test = X[test_indices]
    # ASD
    # 116 = 3 folds for train
    # print(np.shape(X_train))
    # 39 = 1 fold for test
    # print(np.shape(X_test))
    train_channel_means = np.mean(X_train, axis=(0,1,2))
    train_channel_std =   np.std(X_train, axis=(0,1,2))
    return X_train, X_test, train_channel_means, train_channel_std

    
def cast_data(array_of_tensors, subject_type = None, flat_mask = None):
    """
        Casting for NNConv
    """
    N_ROI = array_of_tensors[0].shape[0]
    CHANNELS = array_of_tensors[0].shape[2]
    
    dataset = []
    for mat in array_of_tensors:
            #Allocate numpy arrays 
            edge_index = np.zeros((2, N_ROI * N_ROI))
            edge_attr = np.zeros((N_ROI * N_ROI,CHANNELS))
            x = np.zeros((N_ROI, 1))
            y = np.zeros((1,))
            
            counter = 0
            for i in range(N_ROI):
                for j in range(N_ROI):
                    edge_index[:, counter] = [i, j]
                    edge_attr[counter, :] = mat[i, j]
                    counter += 1
    
            #Fill node feature matrix (no features every node is 1)
            for i in range(N_ROI):
                x[i,0] = 1
                
            #Get graph labels
            y[0] = None
            
            if flat_mask is not None:
                edge_index_masked = []
                edge_attr_masked = []
                for i,val in enumerate(flat_mask):
                    if val == 1:
                        edge_index_masked.append(edge_index[:,i])
                        edge_attr_masked.append(edge_attr[i,:])
                edge_index = np.array(edge_index_masked).T
                edge_attr = edge_attr_masked
            
            
            edge_index = torch.tensor(edge_index, dtype = torch.long)
            edge_attr = torch.tensor(edge_attr, dtype = torch.float)
            x = torch.tensor(x, dtype = torch.float)
            y = torch.tensor(y, dtype = torch.float)
            con_mat = torch.tensor(mat, dtype=torch.float)
            data = Data(x = x, edge_index=edge_index, edge_attr=edge_attr, con_mat = con_mat,  y=y, label = subject_type)
            dataset.append(data)
    return dataset


def save_cbt(img, i, k, fed):
    #Fold i and Client k
    img = np.repeat(np.repeat(img, 10, axis=1), 10, axis=0)
    plt.imshow(img)
    plt.title("Fold " + str(i) + " Client " + str(k))
    plt.axis('off')
    if not os.path.exists('output/' + "CBT_images"):
        os.mkdir('output/' + "CBT_images")
    if not fed:
        if not os.path.exists('output/' + "CBT_images" + "/non-fed"):
            os.mkdir('output/' + "CBT_images" + "/non-fed")
        if not os.path.exists('output/' + "CBT_images" + "/non-fed" + '/' + "Fold " + str(i)):
            os.mkdir('output/' + "CBT_images" + "/non-fed" + '/' + "Fold " + str(i))
        plt.savefig('output/' + "CBT_images" + "/non-fed" + '/' + "Fold " + str(i) + "/" + "Client " + str(k) ,bbox_inches='tight')
    else:
        if not os.path.exists('output/' + "CBT_images" + "/fed"):
            os.mkdir('output/' + "CBT_images" + "/fed")
        if not os.path.exists('output/' + "CBT_images" + "/fed" + '/' + "Fold " + str(i)):
            os.mkdir('output/' + "CBT_images" + "/fed" + '/' + "Fold " + str(i))
        plt.savefig('output/' + "CBT_images" + "/fed" + '/' + "Fold " + str(i) + "/" + "Client " + str(k) ,bbox_inches='tight')
    plt.close()
    
    
# Not used
def show_cbt(img, i, k):
    #Fold i and Client k
    img = np.repeat(np.repeat(img, 10, axis=1), 10, axis=0)
    plt.imshow(img)
    plt.title("Fold " + str(i) + " Client " + str(k))
    plt.axis('off')
    
    
#Clears the given directory
def clear_dir(dir_name):
    for file in os.listdir(dir_name):
        os.remove(os.path.join(dir_name, file))
        

# Load views from a directory with many mat files. Each mat file is a patient's dictionary, containing a ndarray-type view with shape of (35,35,6)
# dir_path should be relative
def load_input_from_dir_of_mats(dir_path):
    views = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        dict = scipy.io.loadmat(file_path)
        view = dict["views"]
        # np.append(views, view)
        views.append(view)
    views = np.asarray(views)
    if not os.path.exists("input"):
            os.makedirs("input")
    np.save("input/" + "dataset", views)


def plotLosses(loss_table_list_non_fed, loss_table_list_fed1, loss_table_list_fed10, loss_table_list_fed1000):
    '''
    This function plots every model's every fold's loss performance and saves them with their particular information written with their names.
    '''
    if not os.path.exists("output/Loss_images"):
        os.makedirs("output/Loss_images")
    shutil.rmtree('output/' + "Loss_images")
    
    non_fed_rep_sum = [0] * config.number_of_clients
    fed_rep_sum1 = [0] * config.number_of_clients
    fed_rep_sum10 = [0] * config.number_of_clients
    fed_rep_sum1000 = [0] * config.number_of_clients
    
    non_fed_kl_sum = [0] * config.number_of_clients
    fed_kl_sum1 = [0] * config.number_of_clients
    fed_kl_sum10 = [0] * config.number_of_clients
    fed_kl_sum1000 = [0] * config.number_of_clients
    
    non_fed_overall_sum = [0] * config.number_of_clients
    fed_overall_sum1 = [0] * config.number_of_clients
    fed_overall_sum10 = [0] * config.number_of_clients
    fed_overall_sum1000 = [0] * config.number_of_clients
    
    for i in range(config.number_of_folds):
        os.makedirs("output/Loss_images/Fold " + str(i))
        cur_loss_table_non_fed = loss_table_list_non_fed[i]
        cur_loss_table_fed1 = loss_table_list_fed1[i]
        cur_loss_table_fed10 = loss_table_list_fed10[i]
        cur_loss_table_fed1000 = loss_table_list_fed1000[i]
        labels = ["Client " + str(i) for i in range(config.number_of_clients)]
        
        # rep_loss
        non_fed_rep = [rep_loss.item() for (rep_loss,_) in cur_loss_table_non_fed]
        fed_rep1 = [rep_loss.item() for (rep_loss,_) in cur_loss_table_fed1]
        fed_rep10 = [rep_loss.item() for (rep_loss,_) in cur_loss_table_fed10]
        fed_rep1000 = [rep_loss.item() for (rep_loss,_) in cur_loss_table_fed1000]
        
        non_fed_rep_sum = list(map(add, non_fed_rep_sum, non_fed_rep))
        fed_rep_sum1 = list(map(add, fed_rep_sum1, fed_rep1))
        fed_rep_sum10 = list(map(add, fed_rep_sum10, fed_rep10))
        fed_rep_sum1000 = list(map(add, fed_rep_sum1000, fed_rep1000))
        lower = min(non_fed_rep+fed_rep1+fed_rep10+fed_rep1000)
        upper = max(non_fed_rep+fed_rep1+fed_rep10+fed_rep1000)
        diff = upper - lower
        lower -= 0.3*diff
        upper += 0.3*diff
        x = np.arange(len(labels))  # the label locations
        width = 0.17  # the width of the bars
        fig, ax = plt.subplots()
        ax.set_ylim(lower, upper)
        fig.set_size_inches(12.8,9.6)
        rects1 = ax.bar(x - 3*width/2, non_fed_rep, width, label='Without federation')
        rects2 = ax.bar(x - width/2, fed_rep1, width, label='f=1')
        rects3 = ax.bar(x + width/2, fed_rep10, width, label='f=10')
        rects4 = ax.bar(x + 3*width/2, fed_rep1000, width, label='f=1000')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Rep loss')
        ax.set_title('Fold ' + str(i) + " Rep loss comparison")
        ax.set_xticks(x, labels)
        ax.legend()
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)
        ax.bar_label(rects4, padding=3)
        fig.tight_layout()
        plt.savefig("output/Loss_images/Fold " + str(i) + "/Rep")
        plt.close()
        # plt.show()
        
        # kl loss
        non_fed_kl = [kl_loss for (_,kl_loss) in cur_loss_table_non_fed]
        fed_kl1 = [kl_loss for (_,kl_loss) in cur_loss_table_fed1]
        fed_kl10 = [kl_loss for (_,kl_loss) in cur_loss_table_fed10]
        fed_kl1000 = [kl_loss for (_,kl_loss) in cur_loss_table_fed1000]
        non_fed_kl_sum = list(map(add, non_fed_kl_sum, non_fed_kl))
        fed_kl_sum1 = list(map(add, fed_kl_sum1, fed_kl1))
        fed_kl_sum10 = list(map(add, fed_kl_sum10, fed_kl10))
        fed_kl_sum1000 = list(map(add, fed_kl_sum1000, fed_kl1000))
        lower = min(non_fed_kl+fed_kl1+fed_kl10+fed_kl1000)
        upper = max(non_fed_kl+fed_kl1+fed_kl10+fed_kl1000)
        diff = upper - lower
        lower -= 0.3*diff
        upper += 0.3*diff
        x = np.arange(len(labels))  # the label locations
        width = 0.17  # the width of the bars
        fig, ax = plt.subplots()
        ax.set_ylim(lower, upper)
        fig.set_size_inches(12.8,9.6)
        rects1 = ax.bar(x - 3*width/2, non_fed_kl, width, label='Without federation')
        rects2 = ax.bar(x - width/2, fed_kl1, width, label='f=1')
        rects3 = ax.bar(x + width/2, fed_kl10, width, label='f=10')
        rects4 = ax.bar(x + 3*width/2, fed_kl1000, width, label='f=1000')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('KL loss')
        ax.set_title('Fold ' + str(i) + " KL loss comparison")
        ax.set_xticks(x, labels)
        ax.legend()
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)
        ax.bar_label(rects4, padding=3)
        fig.tight_layout()
        plt.savefig("output/Loss_images/Fold " + str(i) + "/KL")
        plt.close()
        # plt.show()
        
        # overall loss
        non_fed_overall = [kl_loss * config.PARAMS["lambda_kl"] + rep_loss for (rep_loss,kl_loss) in cur_loss_table_non_fed]
        fed_overall1 = [kl_loss * config.PARAMS["lambda_kl"] + rep_loss for (rep_loss,kl_loss) in cur_loss_table_fed1]
        fed_overall10 = [kl_loss * config.PARAMS["lambda_kl"] + rep_loss for (rep_loss,kl_loss) in cur_loss_table_fed10]
        fed_overall1000 = [kl_loss * config.PARAMS["lambda_kl"] + rep_loss for (rep_loss,kl_loss) in cur_loss_table_fed1000]
        non_fed_overall_sum = list(map(add, non_fed_overall_sum, non_fed_overall))
        fed_overall_sum1 = list(map(add, fed_overall_sum1, fed_overall1))
        fed_overall_sum10 = list(map(add, fed_overall_sum10, fed_overall10))
        fed_overall_sum1000 = list(map(add, fed_overall_sum1000, fed_overall1000))
        lower = min(non_fed_overall+fed_overall1+fed_overall10+fed_overall1000)
        upper = max(non_fed_overall+fed_overall1+fed_overall10+fed_overall1000)
        diff = upper - lower
        lower -= 0.3*diff
        upper += 0.3*diff
        x = np.arange(len(labels))  # the label locations
        width = 0.17  # the width of the bars
        fig, ax = plt.subplots()
        ax.set_ylim(lower, upper)
        fig.set_size_inches(12.8,9.6)
        rects1 = ax.bar(x - 3*width/2, non_fed_overall, width, label='Without federation')
        rects2 = ax.bar(x - width/2, fed_overall1, width, label='f=1')
        rects3 = ax.bar(x + width/2, fed_overall10, width, label='f=10')
        rects4 = ax.bar(x + 3*width/2, fed_overall1000, width, label='f=1000')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Overall loss')
        ax.set_title('Fold ' + str(i) + " overall loss comparison")
        ax.set_xticks(x, labels)
        ax.legend()
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)
        ax.bar_label(rects4, padding=3)
        fig.tight_layout()
        plt.savefig("output/Loss_images/Fold " + str(i) + "/Overall")
        plt.close()
        # plt.show()
        
    # Combining all folds
    os.makedirs("output/Loss_images/Average")
    # rep_loss
    non_fed_rep_average = [x/config.number_of_folds for x in non_fed_rep_sum]
    fed_rep_averge1 = [x/config.number_of_folds for x in fed_rep_sum1]
    fed_rep_averge10 = [x/config.number_of_folds for x in fed_rep_sum10]
    fed_rep_averge1000 = [x/config.number_of_folds for x in fed_rep_sum1000]
    lower = min(non_fed_rep_average+fed_rep_averge1+fed_rep_averge10+fed_rep_averge1000)
    upper = max(non_fed_rep_average+fed_rep_averge1+fed_rep_averge10+fed_rep_averge1000)
    diff = upper - lower
    lower -= 0.3*diff
    upper += 0.3*diff
    x = np.arange(len(labels))  # the label locations
    width = 0.17  # the width of the bars
    fig, ax = plt.subplots()
    ax.set_ylim(lower, upper)
    fig.set_size_inches(12.8,9.6)
    rects1 = ax.bar(x - 3*width/2, non_fed_rep_average, width, label='Without federation')
    rects2 = ax.bar(x - width/2, fed_rep_averge1, width, label='f=1')
    rects3 = ax.bar(x + width/2, fed_rep_averge10, width, label='f=10')
    rects4 = ax.bar(x + 3*width/2, fed_rep_averge1000, width, label='f=1000')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Rep loss')
    ax.set_title("Average Rep loss comparison")
    ax.set_xticks(x, labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    fig.tight_layout()
    plt.savefig("output/Loss_images/Average" + "/Rep")
    plt.close()
    # plt.show()
    
    # kl loss
    non_fed_kl_average = [x/config.number_of_folds for x in non_fed_kl_sum] 
    fed_kl_average1 = [x/config.number_of_folds for x in fed_kl_sum1]
    fed_kl_average10 = [x/config.number_of_folds for x in fed_kl_sum10] 
    fed_kl_average1000 = [x/config.number_of_folds for x in fed_kl_sum1000]
    lower = min(non_fed_kl_average+fed_kl_average1+fed_kl_average10+fed_kl_average1000)
    upper = max(non_fed_kl_average+fed_kl_average1+fed_kl_average10+fed_kl_average1000)
    diff = upper - lower
    lower -= 0.3*diff
    upper += 0.3*diff
    x = np.arange(len(labels))  # the label locations
    width = 0.17  # the width of the bars
    fig, ax = plt.subplots()
    ax.set_ylim(lower, upper)
    fig.set_size_inches(12.8,9.6)
    rects1 = ax.bar(x - 3*width/2, non_fed_kl_average, width, label='Without federation')
    rects2 = ax.bar(x - width/2, fed_kl_average1, width, label='f=1')
    rects3 = ax.bar(x + width/2, fed_kl_average10, width, label='f=10')
    rects4 = ax.bar(x + 3*width/2, fed_kl_average1000, width, label='f=1000')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('KL loss')
    ax.set_title("Average KL loss comparison")
    ax.set_xticks(x, labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    fig.tight_layout()
    plt.savefig("output/Loss_images/Average" + "/KL")
    plt.close()
    # plt.show()
    
    # overall loss
    non_fed_overall_average = [x/config.number_of_folds for x in non_fed_overall_sum] 
    fed_overall_average1 = [x/config.number_of_folds for x in fed_overall_sum1]
    fed_overall_average10 = [x/config.number_of_folds for x in fed_overall_sum10] 
    fed_overall_average1000 = [x/config.number_of_folds for x in fed_overall_sum1000]
    lower = min(non_fed_overall_average+fed_overall_average1+fed_overall_average10+fed_overall_average1000)
    upper = max(non_fed_overall_average+fed_overall_average1+fed_overall_average10+fed_overall_average1000)
    diff = upper - lower
    lower -= 0.3*diff
    upper += 0.3*diff
    x = np.arange(len(labels))  # the label locations
    width = 0.17  # the width of the bars
    fig, ax = plt.subplots()
    ax.set_ylim(lower, upper)
    fig.set_size_inches(12.8,9.6)
    rects1 = ax.bar(x - 3*width/2, non_fed_overall_average, width, label='Without federation')
    rects2 = ax.bar(x - width/2, fed_overall_average1, width, label='f=1')
    rects3 = ax.bar(x + width/2, fed_overall_average10, width, label='f=10')
    rects4 = ax.bar(x + 3*width/2, fed_overall_average1000, width, label='f=1000')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Overall loss')
    ax.set_title("Average overall loss comparison")
    ax.set_xticks(x, labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    fig.tight_layout()
    plt.savefig("output/Loss_images/Average" + "/Overall")
    plt.close()
    # plt.show()
    

# Function overloading for ploting only one loss table
def plotSingleLoss(loss_table):
    if not os.path.exists("output/Loss_images"):
        os.makedirs("output/Loss_images")
    shutil.rmtree('output/' + "Loss_images")
    
    fed_rep_sum1 = [0] * config.number_of_clients
    
    for i in range(config.number_of_folds):
        os.makedirs("output/Loss_images/Fold " + str(i))
        cur_loss_table_fed1 = loss_table[i]
        labels = ["Client " + str(i) for i in range(config.number_of_clients)]
        
        fed_rep1 = [rep_loss.item() for (rep_loss,_) in cur_loss_table_fed1]
        fed_rep_sum1 = list(map(add, fed_rep_sum1, fed_rep1))
        
        lower = min(fed_rep1)
        upper = max(fed_rep1)
        diff = upper - lower
        lower -= 0.3*diff
        upper += 0.3*diff
        x = np.arange(len(labels))  # the label locations
        width = 0.5  # the width of the bars
        fig, ax = plt.subplots()
        ax.set_ylim(lower, upper)
        fig.set_size_inches(12.8,9.6)
        
        rects2 = ax.bar(x, fed_rep1, width, label='f=1')
        
        ax.set_ylabel('Rep loss')
        ax.set_title('Fold ' + str(i) + " Rep loss comparison")
        ax.set_xticks(x, labels)
        ax.legend()
        
        ax.bar_label(rects2, padding=3)
        
        fig.tight_layout()
        plt.savefig("output/Loss_images/Fold " + str(i) + "/Rep")
        plt.close()
        
        
        # Combining all folds
    os.makedirs("output/Loss_images/Average")
    # rep_loss
    fed_rep_averge1 = [x/config.number_of_folds for x in fed_rep_sum1]
    lower = min(fed_rep_averge1)
    upper = max(fed_rep_averge1)
    diff = upper - lower
    lower -= 0.3*diff
    upper += 0.3*diff
    x = np.arange(len(labels))  # the label locations
    width = 0.5  # the width of the bars
    fig, ax = plt.subplots()
    ax.set_ylim(lower, upper)
    fig.set_size_inches(12.8,9.6)
    rects2 = ax.bar(x, fed_rep_averge1, width, label='f=1')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Rep loss')
    ax.set_title("Average Rep loss comparison")
    ax.set_xticks(x, labels)
    ax.legend()
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    plt.savefig("output/Loss_images/Average" + "/Rep")
    plt.close()

def clear_output():
    if not os.path.exists('output/'):
            os.mkdir('output/')
    shutil.rmtree('output/')
    os.mkdir('output/')

def loss_compare_list_init(n_folds, n_clients, n_epochs):
    return np.zeros((n_folds, n_clients, 4, n_epochs))

# option: 0=loss 1=rep_loss 2=kl_loss
def plotLossesCompare(list, option):
    if option == 0:
        if not os.path.exists("output/Loss_vs_Epoch"):
            os.makedirs("output/Loss_vs_Epoch")
        shutil.rmtree('output/' + "Loss_vs_Epoch")
    elif option == 1:
        if not os.path.exists("output/Rep_loss_vs_Epoch"):
            os.makedirs("output/Rep_loss_vs_Epoch")
        shutil.rmtree('output/' + "Rep_loss_vs_Epoch")
    elif option == 2:
        if not os.path.exists("output/KL_loss_vs_Epoch"):
            os.makedirs("output/KL_loss_vs_Epoch")
        shutil.rmtree('output/' + "KL_loss_vs_Epoch")
        
    for i in range(config.number_of_folds):
        if option == 0:
            os.makedirs("output/Loss_vs_Epoch/Fold " + str(i))
        elif option == 1:
            os.makedirs("output/Rep_loss_vs_Epoch/Fold " + str(i))
        elif option == 2:
            os.makedirs("output/KL_loss_vs_Epoch/Fold " + str(i))
            
        for j in range(config.number_of_clients):
            non_fed_losses = list[i][j][0]
            fed_losses1 = list[i][j][1]
            fed_losses10 = list[i][j][2]
            fed_losses1000 = list[i][j][3]
            # Handle early stopping 
            try:
                stop_index_non_fed = np.where(non_fed_losses == 0)[0][0]
            except IndexError as e:
                stop_index_non_fed = config.n_max_epochs
            
            try:
                stop_index_fed1 = np.where(fed_losses1 == 0)[0][0]
            except IndexError as e:
                stop_index_fed1 = config.n_max_epochs
                
            try:
                stop_index_fed10 = np.where(fed_losses10 == 0)[0][0]
            except IndexError as e:
                stop_index_fed10 = config.n_max_epochs
                
            try:
                stop_index_fed1000 = np.where(fed_losses1000 == 0)[0][0]
            except IndexError as e:
                stop_index_fed1000 = config.n_max_epochs
        
            
                
            non_fed_losses = non_fed_losses[0: stop_index_non_fed]
            fed_losses1 = fed_losses1[0: stop_index_fed1]
            fed_losses10 = fed_losses10[0: stop_index_fed10]
            fed_losses1000 = fed_losses1000[0: stop_index_fed1000]
            
            # fed_losses = list[i][j][1]
            # try:
            #     stop_index_fed = np.where(fed_losses == 0)[0][0]
            # except IndexError as e:
            #     stop_index_fed = config.n_max_epochs
            # fed_losses1 = fed_losses1[0: stop_index_fed1]
            
            fig, ax = plt.subplots()
            plt.xlabel("epochs")
            plt.ylabel("loss")
            
            if option == 0:
                plt.title("Fold " + str(i) + " Client " + str(j) + " loss comparison")
            elif option == 1:
                plt.title("Fold " + str(i) + " Client " + str(j) + " rep loss comparison")
            elif option == 2:
                plt.title("Fold " + str(i) + " Client " + str(j) + " kl loss comparison")
            
            # assert len({stop_index_non_fed, stop_index_fed1, stop_index_fed10, stop_index_fed1000}) == 1
            plt.plot(range(0, stop_index_non_fed), non_fed_losses, alpha=0.4, label = "non-fed")
            plt.plot(range(0, stop_index_fed1), fed_losses1, alpha=0.3, label = "f=1")
            plt.plot(range(0, stop_index_fed10), fed_losses10, alpha=0.3, label = "f=10")
            plt.plot(range(0, stop_index_fed1000), fed_losses1000, alpha=0.3, label = "f=1000")
            
            start_from = 0
            end_at = 300
            # plt.plot(range(start_from, stop_index_fed1), fed_losses1[start_from:], "-+", label = "f=1")
            # plt.plot(range(start_from, stop_index_fed10), fed_losses10[start_from:], "-*", label = "f=10")
            # plt.plot(range(start_from, stop_index_fed1000), fed_losses1000[start_from:], "-x", label = "f=1000")
            
            # plt.plot(range(start_from, end_at), non_fed_losses[start_from:end_at], alpha=0.4, label = "non-fed")
            # plt.plot(range(start_from, end_at), fed_losses1[start_from:end_at], alpha=0.4, label = "f=1")
            # plt.plot(range(start_from, end_at), fed_losses10[start_from:end_at], alpha=0.4, label = "f=10")
            # plt.plot(range(start_from, end_at), fed_losses1000[start_from:end_at], alpha=0.4, label = "f=1000")
            
            plt.grid(True)
            plt.legend(bbox_to_anchor = (1.0, 1), loc = 1, borderaxespad = 0.)
            if option == 0:
                plt.savefig("output/Loss_vs_Epoch/Fold " + str(i) + "/Client " + str(j))
            elif option == 1:
                plt.savefig("output/Rep_loss_vs_Epoch/Fold " + str(i) + "/Client " + str(j))
            elif option == 2:
                plt.savefig("output/KL_loss_vs_Epoch/Fold " + str(i) + "/Client " + str(j))

            plt.close()

