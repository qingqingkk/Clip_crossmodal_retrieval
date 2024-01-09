import parser
from train import Fine_Tune, Zero_Shot
import os

if __name__ == '__main__':
    args = parser.parse_arguments()
    task = args.task
    model_dir = args.model_path
    result_dir = args.result_path
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    if task == 'zero_shot':
        Zero_Shot(args)
    elif task == 'fine_tune':
        Fine_Tune(args)