#!/usr/bin/env python3

import torch
import numpy as np
np.float = float
np.bool = bool
np.int = int
import argparse
import random
import os



os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

def _get_og_config(scene_name, obj_config): #config preprossing
    import omnigibson as og
    import yaml
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_name,
            #"load_room_types": ['kitchen', 'living_room'],
            #"load_room_types": ['living_room'], #Workaround: to speed up scene loading
            #"load_room_types": ['kitchen'], #Workaround: to speed up scene loading
            #"trav_map_resolution": 0.01, # meter per pixel, default is 0.1
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": ["rgb"],
                #"default_arm_pose": "diagonal30",
                #"default_reset_mode": "tuck",
                "action_type":"continuous",
                "action_normalize":True,
                "grasping_mode":'physical',
                "position": [0.0, 0.0, 0.0],  # Adjusted position
                #"orientation": [0.9914, 0, 0, 0.1305], #15 deg, Fetch disappeared
                #"orientation": [0.5258, 0, 0, 0.8506], #116 deg
                #"orientation": [0.7071, 0, 0, 0.7071], #90 deg
                #"orientation": [0.8660, 0, 0, 0.5000], #60 deg
                #Bug: The position/orientation modification never works for some reason, rotate the robot as a workaround
                
            },
        ],
        "task":{
            'type': "GraspTask",
            'obj_name': obj_config['name'],
            'objects_config':[obj_config],
        },
        "objects":[
            { #https://github.com/StanfordVL/OmniGibson/blob/main/omnigibson/examples/action_primitives/solve_simple_task.py
                "type": "DatasetObject",
                "name": "cologne",
                "category": "bottle_of_cologne",
                "model": "lyipur",
                "position": [-0.5, -1.2, 0.5],
                "orientation": [0, 0, 0, 1],
            },
        ],
    }

    #replace the robot config with Fetch built-in
    fetch_config_filename = os.path.join(og.example_config_path, "fetch_primitives.yaml")
    fetch_config = yaml.load(open(fetch_config_filename, "r"), Loader=yaml.FullLoader)
    cfg['robots'] = fetch_config['robots']
    #cfg['robots'][0]['position'] = [1.0, 1.0, 0.0]
    return cfg


def get_temp_og_cfg():
    import omnigibson as og
    robot_position = [-1,2.5]
    scene_name = 'Rs_int'
    obj_config = {'pos': [-0.01513892412185669, 1.4189201593399048, 1.0810081958770752], 'ori': [0.5094113349914551, -0.07470039278268814, 0.2355378270149231, -0.824282705783844], 'init_info': {'class_module': 'omnigibson.objects.dataset_object', 'class_name': 'DatasetObject', 'args': {'name': 'wooden_spoon_83', 'category': 'wooden_spoon', 'model': 'aeubta', 'prim_type': 0, 'uuid': 36377017, 'scale': [1.0, 1.0, 1.0], 'in_rooms': ['kitchen_0', 'living_room_0']}}}
    obj_config['category'] = obj_config['init_info']['args']['category']
    obj_config['name'] = obj_config['init_info']['args']['name']
    obj_config['type'] = "DatasetObject"
    cfg = _get_og_config(scene_name, obj_config)
    del cfg['task'] #Bug: The task triggers "Resetting error:  Could not infer dtype of JointPrim", don't know why
    cfg['robots'][0]['position'] = [0,0,0]
    cfg['robots'][0]['position'] = [robot_position[0], robot_position[1], cfg['robots'][0]['position'][2]]
    print(cfg)
    return cfg



#from habitat import logger
import logging
class HabitatLogger(logging.Logger):
    def __init__(
        self,
        name,
        level,
        filename=None,
        filemode="a",
        stream=None,
        format_str=None,
        dateformat=None,
        style="%",
    ):
        super().__init__(name, level)
        if filename is not None:
            handler = logging.FileHandler(filename, filemode)  # type:ignore
        else:
            handler = logging.StreamHandler(stream)  # type:ignore
        self._formatter = logging.Formatter(format_str, dateformat, style)
        handler.setFormatter(self._formatter)
        super().addHandler(handler)

    def add_filehandler(self, log_filename):
        filehandler = logging.FileHandler(log_filename)
        filehandler.setFormatter(self._formatter)
        self.addHandler(filehandler)


logger = HabitatLogger(
    name="habitat",
    level=int(os.environ.get("HABITAT_LAB_LOG", logging.INFO)),
    format_str="%(asctime)-15s %(message)s",
)

#from habitat_baselines.common.baseline_registry import baseline_registry


#import habitat_extensions  # noqa: F401
#import vlnce_baselines  # noqa: F401



# from vlnce_baselines.nonlearning_agents import (
#     evaluate_agent,
#     nonlearning_inference,
# )


def debug_log(msg):
    with open("debug_log.txt", "a") as debug_file:
        debug_file.write(f"run: {msg} \n")




def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        required=True,
        help="experiment id that matches to exp-id in Notion log",
    )
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference"],
        required=True,
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument('--local-rank', type=int, default=0, help="local gpu id")
    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_name: str, exp_config: str, 
            run_type: str, opts=None, local_rank=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """

    #og_test = True
    og_test = False
    if og_test:
        import threading
        env_lock = threading.Lock()
        with env_lock:
            from omnigibson.macros import gm
            import omnigibson as og
            from omnigibson import Environment
            import yaml

            gm.USE_GPU_DYNAMICS = False
            gm.ENABLE_FLATCACHE = True
            og_cfg = get_temp_og_cfg()
            envs = Environment(og_cfg)
    else: 
        envs = None



    from vlnce_baselines.ss_trainer_ETP import RLTrainer
    from vlnce_baselines.config.default import get_config

    config = get_config(exp_config, opts)
    config.defrost()

    config.TENSORBOARD_DIR += exp_name
    config.CHECKPOINT_FOLDER += exp_name
    if os.path.isdir(config.EVAL_CKPT_PATH_DIR):
        config.EVAL_CKPT_PATH_DIR += exp_name
    config.RESULTS_DIR += exp_name
    config.VIDEO_DIR += exp_name
    # config.TASK_CONFIG.TASK.RXR_INSTRUCTION_SENSOR.max_text_len = config.IL.max_text_len
    config.LOG_FILE = exp_name + '_' + config.LOG_FILE

    if 'CMA' in config.MODEL.policy_name and 'r2r' in config.BASE_TASK_CONFIG_PATH:
        config.TASK_CONFIG.DATASET.DATA_PATH = 'data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}.json.gz'

    config.local_rank = local_rank

    #config.EVAL.fast_eval = True
    #debug_log(config)
    config.EVAL.EPISODE_COUNT = 5
    debug_log(f"setting self.config.EVAL.EPISODE_COUNT  s {config.EVAL.EPISODE_COUNT } for testing.")
    

    config.freeze()
    os.system("mkdir -p data/logs/running_log")
    logger.add_filehandler('data/logs/running_log/'+config.LOG_FILE)

    random.seed(config.TASK_CONFIG.SEED)

    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)
    

    # if run_type == "eval" and config.EVAL.EVAL_NONLEARNING:
    #     evaluate_agent(config)
    #     return

    # if run_type == "inference" and config.INFERENCE.INFERENCE_NONLEARNING:
    #     nonlearning_inference(config)
    #     return

    #trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)


    trainer_init = RLTrainer
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    # import pdb; pdb.set_trace()

    debug_log(f"config.EVAL.fast_eval: {config.EVAL.fast_eval}")



    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval(envs)
    elif run_type == "inference":
        trainer.inference()

if __name__ == "__main__":
    main()
