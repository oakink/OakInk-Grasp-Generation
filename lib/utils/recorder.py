import os
import pickle
import random
import sys
import time
from pprint import pformat
from typing import Dict, List, Optional, TypeVar, Union

import numpy as np
import torch
from git import Repo
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from lib.datasets.grasp_query import Queries
from lib.metrics.basic_metric import LossMetric

from .dist_utils import master_only
from .io_utils import (load_model, load_random_state, load_train_param, save_states)
from .logger import logger
from .misc import RandomState
from .net_utils import batch_to_cpu

T = TypeVar("T", bound="Recorder")


class Recorder:

    def __init__(
        self: T,
        exp_id: str,
        cfg: Dict,
        root_path: str = "exp",
        rank: Optional[int] = None,
        time_f: Optional[float] = None,
    ):

        # assert (exp_id == "default" or self.get_git_commit()), "MUST commit before the experiment!"
        self.timestamp = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(time_f if time_f else time.time()))
        self.exp_id = exp_id
        self.cfg = cfg
        self.dump_path = os.path.join(root_path, f"{exp_id}_{self.timestamp}")  #exp/<id>_<timestamp>/
        self.eval_dump_path = os.path.join(self.dump_path, "evaluations")  #exp/<id_timestamp>/evaluations/
        self.tensorboard_path = os.path.join(self.dump_path, "runs")  #exp/<id>_<timestamp>/runs/
        self.rank = rank
        self._record_init_info()

    @master_only
    def _record_init_info(self: T):
        assert self.rank == 0, "Only master process can record init info!"
        if self.rank == 0:  #主进程
            if not os.path.exists(self.dump_path):
                os.makedirs(self.dump_path, exist_ok=True)
            if not os.path.exists(self.eval_dump_path):
                os.makedirs(self.eval_dump_path, exist_ok=True)
            assert logger.filehandler is None, "log file path has been set"
            logger.set_log_file(path=self.dump_path,
                                name=f"{self.exp_id}_{self.timestamp}")  #在path中形成<exp_id>_<timestamp>.log文件
            logger.info(f"run command: {' '.join(sys.argv)}")  #记录本次实验的ran command
            logger.info(f"git commit: {self.get_git_commit()}")  #exp_id是git commit号的话就记录下来
            with open(os.path.join(self.dump_path, "dump_cfg.yaml"), "w") as f:
                f.write(self.cfg.dump(sort_keys=False))  #把本次实验的cfg dump到dump_cfg.yaml中去
                # yaml.dump(self.cfg, f, Dumper=yaml.Dumper, sort_keys=False)
            f.close()
            logger.warning(f"dump cfg file to {os.path.join(self.dump_path, 'dump_cfg.yaml')}")  #记录dump cfg file这件事
        else:
            logger.remove_log_stream()  #其他进程不负责记录
            logger.disabled = True

    @master_only
    def record_checkpoints(self: T, model, optimizer: Union[Dict[str, Optimizer], Optimizer],
                           scheduler: Union[Dict[str, _LRScheduler], _LRScheduler], epoch: int, snapshot: int):
        assert self.rank == 0, "only master process can record checkpoints"
        checkpoints_path = os.path.join(self.dump_path, "checkpoints")  #exp/<id>_<timestamp>/checkpoints/
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)

        # construct RandomState tuple
        random_state = RandomState(
            torch_rng_state=torch.get_rng_state(),
            torch_cuda_rng_state=torch.cuda.get_rng_state(),
            torch_cuda_rng_state_all=torch.cuda.get_rng_state_all(),
            numpy_rng_state=np.random.get_state(),
            random_rng_state=random.getstate(),
        )

        save_states( #在checkpoints/checkpoint/中保存模型文件、随机状态和训练参数
            {
                "epoch": epoch + 1,
                "model": model,
                "optimizer": (optimizer.state_dict()
                              if type(optimizer) is not dict else {k: v.state_dict() for k, v in optimizer.items()}),
                "scheduler": (scheduler.state_dict()
                              if type(scheduler) is not dict else {k: v.state_dict() for k, v in scheduler.items()}),
                "random_state": random_state,
            },
            is_best=False,
            checkpoint=checkpoints_path,
            snapshot=snapshot,
        )
        # logger.info(f"record checkpoints to {checkpoints_path}")

    def resume_checkpoints(self: T,
                           model,
                           optimizer: Union[Dict[str, Optimizer], Optimizer],
                           scheduler: Union[Dict[str, _LRScheduler], _LRScheduler],
                           resume_path: str,
                           resume_epoch: int = -1):
        """[summary]

        Args:
            self (T): [description]
            model ([type]): [description]
            optimizer (Union[Dict[str, Optimizer], Optimizer]): [description]
            scheduler (Union[Dict[str, _LRScheduler], _LRScheduler]): [description]
            resume_path (str): [description]
            resume_epoch (Optional[int], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        map_location = f"cuda:{self.rank}" if self.rank is not None else "cuda"
        resume_path = os.path.join(resume_path, "checkpoints",
                                   f"checkpoint_{resume_epoch}" if resume_epoch != -1 else "checkpoint")
        epoch = load_train_param(optimizer,
                                 scheduler,
                                 os.path.join(resume_path, "train_param.pth.tar"),
                                 map_location=map_location)

        load_random_state(os.path.join(resume_path, "random_state.pkl"))
        load_model(model, resume_path, map_location=map_location)
        return epoch

    @master_only
    def record_loss(self, loss_metric: LossMetric, epoch: int, comment=""):
        assert self.rank == 0, "only master process can record loss"
        loss_dump_path = os.path.join(self.eval_dump_path, f"{comment}_Loss.txt")
        with open(loss_dump_path, "a") as f:
            f.write(f"Epoch {epoch} | {comment} loss metric:\n {pformat(loss_metric.get_measures())}\n\n")

    @master_only
    def record_metric(self, metrics: List, epoch: int, comment=""):
        assert self.rank == 0, "only master process can record metirc"
        metric_dump_path = os.path.join(self.eval_dump_path, f"{comment}_Metric.txt")
        with open(metric_dump_path, "a") as f:
            f.write(f"Epoch {epoch} | {comment} metric:\n")
            for M in metrics:
                f.write(f"{pformat(M.get_measures())}\n")
            f.write("\n")

    @master_only
    def record_batch_result(self, resutls, epoch: int, step_idx: int, comment=""):
        assert self.rank == 0, "only master process can record metirc"
        assert Queries.SAMPLE_IDENTIFIER in resutls, "you must have key: `sample_identifier` to record each results"
        res_dump_path = os.path.join(self.eval_dump_path, f"{comment}_results", f"ep_{epoch}")
        os.makedirs(res_dump_path, exist_ok=True)

        resutls = batch_to_cpu(resutls)
        sample_identifier = resutls[Queries.SAMPLE_IDENTIFIER]

        for i, sample_id in enumerate(sample_identifier):
            sample_res = {}
            for k, v in resutls.items():
                sample_res[k] = v[i]

            with open(os.path.join(res_dump_path, f"{sample_id}.pkl"), "wb") as f:
                pickle.dump(sample_res, f)

    @staticmethod
    def get_git_commit() -> Optional[str]:
        # get current git report
        proj_root = os.environ.get("PROJECT_ROOT")
        if proj_root is not None:
            repo = Repo(proj_root)
        else:
            repo = Repo(".")

        modified_files = [item.a_path for item in repo.index.diff(None)]
        staged_files = [item.a_path for item in repo.index.diff("HEAD")]
        untracked_files = repo.untracked_files

        if len(modified_files):
            logger.error(f"modified_files: {' '.join(modified_files)}")
        if len(staged_files):
            logger.error(f"staged_files: {' '.join(staged_files)}")
        if len(untracked_files):
            logger.error(f"untracked_files: {' '.join(untracked_files)}")

        # return (repo.head.commit.hexsha
        #         if not (len(modified_files) or len(staged_files) or len(untracked_files)) else None)
        return repo.head.commit.hexsha
