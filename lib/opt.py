import argparse

parser = argparse.ArgumentParser(description="Grasp Generation")
"----------------------------- Experiment options -----------------------------"
parser.add_argument("-c", "--cfg", help="experiment configure file name", type=str, default=None)
parser.add_argument("--exp_id", default="default", type=str, help="Experiment ID")
parser.add_argument("--resume", help="resume training from exp", type=str, default=None)
parser.add_argument("--resume_epoch", help="resume from the given epoch", type=int, default=-1)
parser.add_argument("--reload", help="reload checkpoint for test", type=str, default=None)
parser.add_argument("-w", "--workers", help="worker number on each device (default: 0)", type=int, default=0)
parser.add_argument("-b",
                    "--batch_size",
                    help="input batch size on each device (if not specify, will use the one in cfg file)",
                    type=int,
                    default=None)
parser.add_argument("--val_batch_size",
                    help="batch size when val or test, if not specified, will use batch_size",
                    type=int,
                    default=None)
"----------------------------- General options -----------------------------"
parser.add_argument("-g", "--gpu_id", type=str, default=0, help="override enviroment var CUDA_VISIBLE_DEVICES")
parser.add_argument("--snapshot", default=10, type=int, help="How often to take a snapshot of the model (0 = never)")
parser.add_argument("--eval_freq", default=5, type=int, help="How often to evaluate the model on val set")
parser.add_argument("--log_freq", default=10, type=int, help="How often to write summary logs")
"----------------------------- Distributed options -----------------------------"
parser.add_argument("--dist_master_addr", type=str, default="localhost")
parser.add_argument("-p", "--dist_master_port", type=str, default="60001")


def parse_exp_args():
    arg, custom_arg_string = parser.parse_known_args()
    return arg, custom_arg_string
