import torch
import pickle
from argparse import ArgumentParser
import os
from train import Trainer_iwslt_ende, Trainer_wmt_ende
from train_same_ca import Trainer_wmt_ende_multi
from utils.config import Config
import json
import sys

from utils.utils_func import save_data, make_dataset_path



def main(config_path:Config, args:ArgumentParser):
    device = torch.device('cuda:0') if args.device == 'gpu' else torch.device('cpu')
    print('Using {}'.format(device))

    if (args.cont and args.mode == 'train') or args.mode != 'train':
        try:
            config = Config(config_path)
            config = Config(config.base_path + '/model/' + args.name + '/' + args.name + '.json')
            base_path = config.base_path
            data_type = config.data_type
        except:
            print('*'*36)
            print('There is no [-n, --name] argument')
            print('*'*36)
            sys.exit()
    else:
        config = Config(config_path)
        base_path = config.base_path
        data_type = config.data_type

        # make neccessary folders
        os.makedirs(base_path+'model', exist_ok=True)
        os.makedirs(base_path+'loss', exist_ok=True)
        os.makedirs(base_path+'data/wmt14-en-de/processed', exist_ok=True)
        os.makedirs(base_path+'data/iwslt14-en-de/processed', exist_ok=True)
        os.makedirs(base_path+'data/iwslt14-en-de/tokenizer', exist_ok=True)

        # save nmt data
        assert data_type in ['iwslt14-ende', 'wmt14-ende']
        save_data(base_path, data_type)

        # check tokenizer file
        if data_type == 'iwslt14-ende':
            if not os.path.isdir(base_path+'data/iwslt14-en-de/tokenizer/vocab_'+str(config.vocab_size)):
                print('You must make vocab and tokenizer first..')
                sys.exit()
            config.tokenizer_path = base_path+'data/iwslt14-en-de/tokenizer/vocab_'+str(config.vocab_size)+'/vocab.txt'

        # split dataset path
        config.dataset_path = make_dataset_path(base_path, data_type)
        
        # redefine config
        config.loss_data_path = base_path + 'loss/' + config.loss_data_name + '.pkl'

        # make model related files and folder
        model_folder = base_path + 'model/' + config.model_name
        config.model_path = model_folder + '/' + config.model_name + '.pt'
        model_json_path = model_folder + '/' + config.model_name + '.json'
        os.makedirs(model_folder, exist_ok=True)
          
        with open(model_json_path, 'w') as f:
            json.dump(config.__dict__, f)
    
    if config.data_type == 'iwslt14-ende':
        trainer = Trainer_iwslt_ende(config, device, args.mode, args.cont)
    elif config.data_type == 'wmt14-ende':
        if config.DDP ==0:
            trainer = Trainer_wmt_ende(config, device, args.mode, args.cont)
        else:
            trainer = Trainer_wmt_ende_multi(config, args.mode, args.cont)

    if args.mode == 'train':
        loss_data_path = config.loss_data_path
        print('Start training...\n')
        loss_data = trainer.training()

        print('Saving the loss related data...')
        with open(loss_data_path, 'wb') as f:
            pickle.dump(loss_data, f)
            
    elif args.mode == 'inference':
        trainer.inference('test', config.result_num)

    elif args.mode == 'multi_bleu_perl':
        trainer.multi_bleu_perl('test')

    else:
        print("Please select mode among 'train', 'inference', and 'multi_bleu_perl'..")
        sys.exit()



if __name__ == '__main__':
    path = os.path.realpath(__file__)
    path = path[:path.rfind('/')+1] + 'config.json'    

    parser = ArgumentParser()
    parser.add_argument('-d', '--device', type=str, required=True, choices=['cpu', 'gpu'])
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'inference', 'multi_bleu_perl'])
    parser.add_argument('-c', '--cont', type=int, default=0, required=False)
    parser.add_argument('-n', '--name', type=str, required=False)
    args = parser.parse_args()

    main(path, args)