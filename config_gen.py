from ml_collections import ConfigDict
import argparse


def get_config():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()

    parser.add_argument('--global_save_label', type=str, help='label for saving images')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--minibatch_size", type=int, default=4)
    parser.add_argument("--tester", type=str2bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    # parser.add_argument("--prompt", type=str, default='a happy dog')
    parser.add_argument("--n_images", type = int, default=10)
    parser.add_argument("--animals_to_generate", type=str, default='all')
    parser.add_argument("--locations_to_generate", type=str, default='all')
    parser.add_argument("--locations_to_avoid", type=str, default='None')

    # Generation
    parser.add_argument("--seed", type=int, default=10)

    parser.add_argument("--machine_name", type=str, default='bigboy')
    parser.add_argument("--num_iters", type=int, default=4)

    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--diffusion_batch_size", type=int, default=6)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--add_prompt_suffix", type=str2bool, default=True)

    # Data
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--dataset_path", type=str, default="~/data/")
    parser.add_argument("--classification_batch_size", type=int, default=8)

    # Splits of test, train and val
    parser.add_argument("--location_ratio_split", type=str, default='1111', 
                help = "ratio split of location concepts in image dataset")
    parser.add_argument("--day-night_ratio_split", type=str, default='11')
    parser.add_argument("--object_ratio_split", type=str, default='111111')

    # Logging and Saving
    parser.add_argument("--wandb", type=str2bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="PAE")
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="./save/")
    # Classifier
    parser.add_argument("--classifier_arch", type=str, default="resnet50")

    # start idx
    # parser.add_argument("--class_start_idx", type=int, default=0)
    # parser.add_argument("--class_end_idx", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10)
    parser.add_argument("--samples_per_class", type=int, default=1)

    # Output
    parser.add_argument("--generated_images_path", type=str, default="./gen_images/")
    parser.add_argument("--pretrained_model_path", type=str, default="~/models/")
    config = ConfigDict(vars(parser.parse_args()))

    return config
