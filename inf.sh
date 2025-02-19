#hm3d
#./run.py --exp_name release_r2r --run-type eval --exp-config run_r2r/iter_train.yaml TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter12000.pth IL.back_algo control
#og
#OMNIGIBSON_HEADLESS=1 ./run.py --exp_name release_r2r --run-type eval --exp-config run_r2r/iter_train.yaml TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter12000.pth IL.back_algo control > og.log 2>&1
./run.py --exp_name release_r2r --run-type eval --exp-config run_r2r/iter_train.yaml TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter12000.pth IL.back_algo control > og.log 2>&1
#gdb --args python run.py --exp_name release_r2r --run-type eval --exp-config run_r2r/iter_train.yaml TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r/ckpt.iter12000.pth IL.back_algo control > og.log 2>&1


#python -u run_grasp_task.py > output.log 2>&1