def debug_log(msg):
    with open("debug_log.txt", "a") as debug_file:
        debug_file.write(f"simulator_adapter: {msg} \n")

class SimulatorAdapter:
    def __init__(self, config, simulator_type="Habitat"):
        """
        Adapter class to abstract the interaction with simulators like Habitat and OmniGibson.
        Supports switching between different simulators using a common interface.

        Args:
            config (Config): Configuration object for the simulator.
            simulator_type (str): Type of simulator, either 'Habitat' or 'OmniGibson'.
        """
        self.config = config
        self.simulator_type = simulator_type.lower()
        
        if self.simulator_type == "habitat":
            from habitat_baselines.common.environments import get_env_class
            from vlnce_baselines.common.env_utils import construct_envs
            self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME), auto_reset_done=False)

            #self.num_envs = self.envs.num_envs
            #self.number_of_episodes = self.envs.number_of_episodes
            #self.observation_spaces = self.envs.observation_spaces
            #self.action_spaces = self.envs.action_spaces

        elif self.simulator_type == "omnigibson":
            import omnigibson as og
            self.envs = og.Environment(self.config)  # Placeholder for actual OG setup
        else:
            raise ValueError(f"Unsupported simulator type: {self.simulator_type}")

    def get_num_envs(self):
        if self.simulator_type == "habitat":
            return self.envs.num_envs
        elif self.simulator_type == "omnigibson":
            pass
        else:
            raise NotImplementedError
    
    def get_number_of_episodes(self):
        if self.simulator_type == "habitat":
            return self.envs.number_of_episodes
        elif self.simulator_type == "omnigibson":
            pass
        else:
            raise NotImplementedError
    
    def get_observation_spaces(self):
        if self.simulator_type == "habitat":
            return self.envs.observation_spaces
        elif self.simulator_type == "omnigibson":
            pass
        else:
            raise NotImplementedError
    
    def get_action_spaces(self):
        if self.simulator_type == "habitat":
            return self.envs.action_spaces 
        elif self.simulator_type == "omnigibson":
            pass
        else:
            raise NotImplementedError
    
    def reset(self):
        """
        Resets the environment and returns the initial observation.
        """
        if self.simulator_type == "habitat":
            ret = self.envs.reset()
            #debug_log(f"reset len(ret[0]): {len(ret[0])}") #25
            #debug_log(f"reset type(ret[0][0]): {type(ret[0][0])}")
            return ret
        elif self.simulator_type == "omnigibson":
            return self.envs.reset()
        else:
            raise NotImplementedError

    def step(self, actions):
        """
        Steps the environment with the given actions.

        Args:
            actions (list or dict): Actions for each environment instance.

        Returns:
            ret[0][0]:
                Tuple (observations, rewards, dones, infos)
                    observations: 
        """
        if self.simulator_type == "habitat":
            ret = self.envs.step(actions)
            #https://github.com/facebookresearch/habitat-sim/blob/main/src_python/habitat_sim/simulator.py
            #get_sensor_observations


            debug_log(f"type(ret[0][0]), {type(ret[0][0])}") #h-Observations
            debug_log(f"ret[0][0].keys(), {ret[0][0].keys()}") #h-Observations
            #dict_keys(['rgb', 'depth', 'rgb_30', 'rgb_60', 'rgb_90', 'rgb_120', 'rgb_150', 'rgb_180', 'rgb_210', 'rgb_240', 'rgb_270', 'rgb_300', 'rgb_330', 'depth_30', 'depth_60', 'depth_90', 'depth_120', 'depth_150', 'depth_180', 'depth_210', 'depth_240', 'depth_270', 'depth_300', 'depth_330', 'instruction']) 
            debug_log(f"type(ret[0][1]), {type(ret[0][1])}") #float
            debug_log(f"type(ret[0][2]), {type(ret[0][2])}") #bool, Task done?
            debug_log(f"type(ret[0][3]), {type(ret[0][3])}") #h-Metrics
            observations, _, dones, infos = [list(x) for x in zip(*ret)]


            #debug_log(f"step len(ret[0]): {len(ret[0])}") 4
            #debug_log(f"step type(ret[0][0]): {type(ret[0][0])}")
            #debug_log(f"step type(observations): {type(observations)}") #list
            #debug_log(f"step len(observations): {len(observations)}") 1
            #debug_log(f"step observations[0]: {(observations[0])}")
            #raise
            return ret
        elif self.simulator_type == "omnigibson":
            return self.envs.step(actions)
        else:
            raise NotImplementedError
        
    def call_at(self, idx, instr, input_dict=None):
        if self.simulator_type == "habitat":
            if input_dict:
                return self.envs.call_at(idx, instr, input_dict)
            else:
                return self.envs.call_at(idx, instr)
        elif self.simulator_type == "omnigibson":
            pass #placeholder
        else:
            raise NotImplementedError
        #_teacher_action envs.call_at(j, "cand_dist_to_goal", {"angle": angle_k, "forward": forward_k})
        #_teacher_action_new envs.call_at(i, "current_dist_to_goal")
        # envs.call_at(i, "point_dist_to_goal", {"pos": p[1]})
        # envs.call_at(i, "ghost_dist_to_ref", {"ghost_vp_pos": ghost_vp_pos,"ref_path": self.gt_data[str(cur_episodes[i].episode_id)]['locations'],})
        # rollout envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
    def call(self, instr, kargs=None):
        if self.simulator_type == "habitat":
            if kargs:
                return self.envs.call(instr, kargs)
            else:
                return self.envs.call(instr)
        elif self.simulator_type == "omnigibson":
            pass #placeholder
        else:
            raise NotImplementedError
        #_teacher_action envs.call(["get_cand_idx"]*self.envs.num_envs, kargs)
        #get_pos_ori envs.call(['get_pos_ori']*self.envs.num_envs)

    def pause_at(self, index):
        """
        Pauses an environment at the given index.
        Useful for managing multiple environments in parallel.

        Args:
            index (int): Index of the environment to pause.
        """
        if self.simulator_type == "habitat":
            return self.envs.pause_at(index)
        elif self.simulator_type == "omnigibson":
            # OmniGibson does not have pause, so you might simulate it by disabling updates.
            pass  # Placeholder
        else:
            raise NotImplementedError

    def resume_all(self):
        """
        Resumes all paused environments.
        """
        if self.simulator_type == "habitat":
            return self.envs.resume_all()
        elif self.simulator_type == "omnigibson":
            pass  # Placeholder
        else:
            raise NotImplementedError

    def get_current_metrics(self):
        """
        Retrieves current environment metrics such as distance to goal or success rate.

        Returns:
            dict: Metrics dictionary.
        """
        if self.simulator_type == "habitat":
            return self.envs.get_metrics()
        elif self.simulator_type == "omnigibson":
            return self.envs.get_metrics()
        else:
            raise NotImplementedError

    def apply_obs_transforms(self, obs):
        """
        Applies observation transformations like resizing, cropping, or normalization.

        Args:
            obs (dict): Observations.

        Returns:
            dict: Transformed observations.
        """
        if self.simulator_type == "habitat":
            from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
            return apply_obs_transforms_batch(obs, self.config.obs_transforms)
        elif self.simulator_type == "omnigibson":
            # Placeholder: Implement observation transforms for OmniGibson
            return obs
        else:
            raise NotImplementedError

    def current_episodes(self):
        if self.simulator_type == "habitat":
            return self.envs.current_episodes()
        elif self.simulator_type == "omnigibson":
            # Placeholder: Implement observation transforms for OmniGibson
            pass
        else:
            raise NotImplementedError
        
        #current

    def close(self):
        """
        Closes the environment to free up resources.
        """
        self.envs.close()
