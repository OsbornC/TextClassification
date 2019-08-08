import argparse,os
import configparser
def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings
    
    parser.add_argument('--config', type=str, default="no_file_exists",
                    help='gpu number')
        
        
    parser.add_argument('--hidden_dim', type=int, default=128,
                    help='hidden_dim')     

    parser.add_argument('--max_seq_len', type=int, default=200,
                    help='max_seq_len')
    parser.add_argument('--batch_size', type=int, default=32,
                    help='batch_size')
    parser.add_argument('--embedding_dim', type=int, default=300,
                    help='embedding_dim')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                    help='learning_rate')
    parser.add_argument('--grad_clip', type=float, default=1e-1,
                    help='grad_clip')

    parser.add_argument('--model', type=str, default="bert",
                    help='model name')

    parser.add_argument('--dataset', type=str, default="imdb",

                    help='dataset')
    parser.add_argument('--position', type=bool, default=False,
                    help='gpu number')
    
    parser.add_argument('--keep_dropout', type=float, default=0.3,
                    help='keep_dropout')
    parser.add_argument('--max_epoch', type=int, default=20,
                    help='max_epoch')
    parser.add_argument('--embedding_file', type=str, default="glove.6b.300",
                    help='glove or w2v')
    parser.add_argument('--embedding_training', type=str, default="false",
                    help='embedding_training')
    #kim CNN
    parser.add_argument('--kernel_sizes', type=str, default="2,3,4",
                    help='kernel_sizes')
    parser.add_argument('--kernel_nums', type=str, default="128,128,128",
                    help='kernel_nums')
    parser.add_argument('--embedding_type', type=str, default="non-static",
                    help='embedding_type')
    parser.add_argument('--lstm_mean', type=str, default="mean",# last
                    help='lstm_mean')
    parser.add_argument('--lstm_layers', type=int, default=1,# last
                    help='lstm_layers')
    parser.add_argument('--gpu', type=int, default=0,
                    help='gpu number')
    parser.add_argument('--proxy', type=str, default="null",
                    help='http://proxy.xx.com:8080')
    parser.add_argument('--debug', type=str, default="true",
                    help='gpu number')

    parser.add_argument('--embedding_dir', type=str, default=".glove/glove.6B.300d.txt",
                    help='embedding_dir')
    
    parser.add_argument('--bert_dir', type=str, default="D:/dataset/bert/uncased_L-12_H-768_A-12",
                    help='bert dir')
    
    parser.add_argument('--from_torchtext', type=str, default="false",
                    help='from torchtext or native data loader')
#
    parser.add_argument('--shuffle', type=str, default="sequential",
                    help='random shuffle or sequential shuffle')

    parser.add_argument('--datas', type=str, default="first",
                    help='which round for loading data')

    args = parser.parse_args()
    
    if args.config != "no_file_exists":
        if os.path.exists(args.config):
            config = configparser.ConfigParser()
            config_file_path=args.config
            config.read(config_file_path)
            config_common = config['COMMON']
            for key in config_common.keys():
                args.__dict__[key]=config_common[key]
        else:
            print("config file named %s does not exist" % args.config)

    args.kernel_sizes = [int(i) for i in args.kernel_sizes.split(",")]
    args.kernel_nums = [int(i) for i in args.kernel_nums.split(",")]
#
#    # Check if args are valid
#    assert args.rnn_size > 0, "rnn_size should be greater than 0"

    if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
        os.environ["CUDA_VISIBLE_DEVICES"] =str(args.gpu)
    
    if args.model=="transformer":
        args.position=True
    else:
        args.position=False
    if args.debug.lower() =="true":
        args.debug = True
    else:
        args.debug = False
    
    if args.embedding_training.lower() =="true":
        args.embedding_training = True
    else:
        args.embedding_training = False
    if args.from_torchtext.lower() =="true":
        args.from_torchtext = True
    else:
        args.from_torchtext = False
    
    # if args.dataset == 'imdb':
    #     args.max_seq_len = 100
        
    if os.path.exists("proxy.config"):
        with open("proxy.config") as f:

            args.proxy = f.read()
            print(args.proxy)
        

        
    return args 
