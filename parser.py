import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Training parameters
    parser.add_argument("--task", type=str, default='zero_shot', choices=['zero_shot', 'fine_tune']
                        help="define the task")

    parser.add_argument("--data_type", type=str, default='raw', choices=['raw', 'encoded']
                        help="the data is image&text or encoded data")

    parser.add_argument("--bz", type=int, default=256, 
                        help='baech_size')

    parser.add_argument("--model_version", type=str, default='Vit-B/32', choices=['Vit-B/32', 'Vit-L/16']
                        help="version of clip model")

    parser.add_argument("--train_mode", type=str, default='only_proj', choices=['only_proj', 'with_adapter', 'total']
                        help="train only projection layer or total structure")

    parser.add_argument("--optimizer", type=str, default='AdamW', choices=['sgd', 'Adam', 'AdamW']
                        help="type of optimizer")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")

    parser.add_argument("--scheduler", type=str, default=None, choices = ['StepLR', 'ReduceLROnPlateau']
                        help="type of scheduler")

    parser.add_argument("--loss_type", type=str, default='cos_embedd', choices = ['corss_entropy', 'contrastive', 'info_nce_loss', 'nt_xent', 'cos_embedd', 'mix']
                        help="type of loss")               

    parser.add_argument("--max_epochs", type=int, default=20,
                        help="stop when training reaches max_epochs")

    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of processes to use for data loading / preprocessing")

    parser.add_argument("--dataset", type=str, default='moscoco', choices = ['moscoco', 'flickr']
                        help="type of dataset")

    parser.add_argument("--resume", action='store_true',
                        help="resume training")

    parser.add_argument("--sr", action='store_true', 
                        help='save the result or not')


    # Paths parameters
    parser.add_argument("--data_path", type=str, default="/kaggle/input/coco-2017-dataset/coco2017/",
                        help="path to dataset")

    parser.add_argument("--model_path", type=str, default="./model",
                        help="path to val set (must contain database and queries)")

    parser.add_argument("--result_path", type=str, default="./result",
                        help="path to test set (must contain database and queries)")
    
    args = parser.parse_args()
    return args