from utils import create_DFdata_lists
from utils import create_data_lists

if __name__ == '__main__':
    #Public dataset
    #train
    create_data_lists(input_folders='COISR/datasets/COCP-SR/train/Catadioptric',
                      output_folder='COISR/datasets/COCP-SR',
                      split='train')
    #test
    create_data_lists(input_folders='COISR/datasets/COCP-SR/test/Catadioptric',
                      output_folder='COISR/datasets/COCP-SR',
                      split='test')
    #validation
    create_data_lists(input_folders='COISR/datasets/COCP-SR/validation/Catadioptric',
                      output_folder='COISR/datasets/COCP-SR',
                      split='validation')


    #Simulation dataset（DIV2K-Flickr2K）
    #train
    create_DFdata_lists(input_folders=['COISR/datasets/DIV2K_train_HR',
                                       'COISR/datasets/Flickr2K_HR'],
                        min_size_h=480,
                        min_size_w=1696,
                        output_folder='COISR/datasets/SIM-SR',
                        split='train')
    #val
    create_DFdata_lists(input_folders=['COISR/datasets/DIV2K_valid_HR'],
                        min_size_h=480,
                        min_size_w=1696,
                        output_folder='COISR/datasets/SIM-SR',
                        split='validation')
