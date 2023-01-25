from model import MGN_NET
import config

MGN_NET.train_model(n_max_epochs = config.n_max_epochs, 
                        data_path = config.DATASET_PATH, 
                        early_stop=True,
                        model_name = "MGN_NET")