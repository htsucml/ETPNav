

def debug_log(msg):
    with open("debug_log.txt", "a") as debug_file:
        debug_file.write(f"simulator_adapter: {msg} \n")
        

class OGEpisode:
    def __init__(self):
        self.episode_id = 48763 #random placeholder 


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
            '''
            import omnigibson as og
            import yaml
            from omnigibson.macros import gm
            gm.USE_GPU_DYNAMICS = False
            gm.ENABLE_FLATCACHE = True
            '''

            #self.config = yaml.load(open(self.config, "r"), Loader=yaml.FullLoader)
            debug_log(f"omnigibson config: {self.config}")
            self.envs = config #directly pass og envs
            print("started omnigibson in simulationadapter")

            from gym.spaces import Dict, Box, Discrete
            import numpy as np

            self.observation_spaces = [Dict({
                'depth': Box(0.0, 1.0, (256, 256, 1), dtype=np.float32),
                'depth_120': Box(0.0, 1.0, (256, 256, 1), dtype=np.float32),
                'depth_150': Box(0.0, 1.0, (256, 256, 1), dtype=np.float32),
                'depth_180': Box(0.0, 1.0, (256, 256, 1), dtype=np.float32),
                'depth_210': Box(0.0, 1.0, (256, 256, 1), dtype=np.float32),
                'depth_240': Box(0.0, 1.0, (256, 256, 1), dtype=np.float32),
                'depth_270': Box(0.0, 1.0, (256, 256, 1), dtype=np.float32),
                'depth_30': Box(0.0, 1.0, (256, 256, 1), dtype=np.float32),
                'depth_300': Box(0.0, 1.0, (256, 256, 1), dtype=np.float32),
                'depth_330': Box(0.0, 1.0, (256, 256, 1), dtype=np.float32),
                'depth_60': Box(0.0, 1.0, (256, 256, 1), dtype=np.float32),
                'depth_90': Box(0.0, 1.0, (256, 256, 1), dtype=np.float32),
                'instruction': Discrete(4),
                'rgb': Box(0, 255, (224, 224, 3), dtype=np.uint8),
                'rgb_120': Box(0, 255, (224, 224, 3), dtype=np.uint8),
                'rgb_150': Box(0, 255, (224, 224, 3), dtype=np.uint8),
                'rgb_180': Box(0, 255, (224, 224, 3), dtype=np.uint8),
                'rgb_210': Box(0, 255, (224, 224, 3), dtype=np.uint8),
                'rgb_240': Box(0, 255, (224, 224, 3), dtype=np.uint8),
                'rgb_270': Box(0, 255, (224, 224, 3), dtype=np.uint8),
                'rgb_30': Box(0, 255, (224, 224, 3), dtype=np.uint8),
                'rgb_300': Box(0, 255, (224, 224, 3), dtype=np.uint8),
                'rgb_330': Box(0, 255, (224, 224, 3), dtype=np.uint8),
                'rgb_60': Box(0, 255, (224, 224, 3), dtype=np.uint8),
                'rgb_90': Box(0, 255, (224, 224, 3), dtype=np.uint8)
            })]

            from gym.spaces import Dict
            from habitat.core.spaces import ActionSpace, EmptySpace
            action_space = {
                "HIGHTOLOWEVAL": EmptySpace(),
                "MOVE_FORWARD": EmptySpace(),
                "STOP": EmptySpace(),
                "TURN_LEFT": EmptySpace(),
                "TURN_RIGHT": EmptySpace(),
            }
            self.action_space = [ActionSpace(action_space)]
            self.episodes = [OGEpisode()]


            #self.envs = og.Environment(self.config)  # Placeholder for actual OG setup

            '''
            for _ in range(500):
                print(f"action_space: self.envs.action_space")
                action = self.envs.action_space.sample()
                obs, rew, terminated, truncated, info = self.envs.step(action)
                print(f"{_} step: Action:{action}, Obs:{obs}, Rew:{rew}, Terminated:{terminated}, Truncated: {truncated}, Info:{info}")
            raise
            '''
        else:
            raise ValueError(f"Unsupported simulator type: {self.simulator_type}")

    def get_num_envs(self):
        if self.simulator_type == "habitat":
            return self.envs.num_envs
        elif self.simulator_type == "omnigibson":
            return 1
        else:
            raise NotImplementedError
    
    def get_number_of_episodes(self):
        if self.simulator_type == "habitat":
            return self.envs.number_of_episodes
        elif self.simulator_type == "omnigibson":
            return [1]
        else:
            raise NotImplementedError
    
    def get_observation_spaces(self):
        if self.simulator_type == "habitat":
            print(f"self.envs.observation_spaces: {self.envs.observation_spaces}")
            #print(f"self.envs.observation_spaces['rgb_210']: {self.envs.observation_spaces['rgb_210']}")
            #print(f"self.envs.observation_spaces['rgb_210']: {self.envs.observation_spaces['rgb_210']}")
            return self.envs.observation_spaces
        elif self.simulator_type == "omnigibson":            

            return self.observation_spaces
        else:
            raise NotImplementedError
    
    def get_action_spaces(self):
        if self.simulator_type == "habitat":
            print(f"self.envs.action_spaces: {self.envs.action_spaces}")
            return self.envs.action_spaces 
        elif self.simulator_type == "omnigibson":
            return self.action_space
        else:
            raise NotImplementedError
    
    
    def _get_observations(self, keys=None):
        assert self.simulator_type == "omnigibson"
        if keys is None:
            keys = self.get_observation_spaces()[0].keys()
        observations = {k:None for k in keys}

        from omnigibson.sensors import VisionSensor
        from omnigibson.utils.transform_utils import euler2quat, quat_multiply
        import torch
        import numpy as np
        import cv2
        import omnigibson as og
        sim = og.sim
        robot = self.envs.robots[0]
        viewer_camera = og.sim.viewer_camera
        print(f"sim viewer resolution: {sim.viewer_width}x{sim.viewer_height}")

        def process_image(rgb_obs):
            print(f"rgb_obs {rgb_obs}")
            rgb_image = np.array(rgb_obs).astype(np.uint8)
            #rgb_image = np.array(rgb_obs * 255).astype(np.uint8)
            return rgb_image

        def get_pseudo_pano(robot, viewer_camera):
            #yaw_angles = np.arange(0, 360, 30)  # [0, 30, 60, ..., 330]
            robot_pos = robot.get_position()
            quat_init = viewer_camera.get_position_orientation()[1]
            #pano_images = [process_image(viewer_camera.get_obs()[0]["rgb"])]
            pano_images = []
            sim.step()
            for yaw_idx in range(13):
                yaw = 30 + yaw_idx*30
                #viewer_camera.set_position()  # Adjust height
                robot_position = np.array([robot_pos[0], robot_pos[1], 1.5])
                inp = torch.tensor([0, 0, np.radians(yaw)], dtype=torch.float32)
                quat = quat_multiply((euler2quat(inp)), quat_init)
                viewer_camera.set_position_orientation(position=robot_position, orientation=quat)
                sim.step()
                print(f"camera {yaw} {viewer_camera.get_position(), viewer_camera.get_orientation()}")

                # Capture image
                rgb_obs = viewer_camera.get_obs()[0]["rgb"]
                rgb_image = process_image(rgb_obs)
                pano_images.append(rgb_image)
            pano_images = pano_images[1:]
            #This is a workaround of an unknown bug causing the first frame = default view frame
            return pano_images

        pano_images = get_pseudo_pano(robot, viewer_camera)
        stitched_image = cv2.hconcat(pano_images) 
        output_path = "test_og_camera_in_sa_pano_3.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR))
        for idx, angle in enumerate(range(0, 360, 30)):
            k = f'rgb_{angle}' if angle!=0 else 'rgb'
            observations[k] = pano_images[idx]
            k = f'depth_{angle}' if angle!=0 else 'depth'
            print("WARNING: _get_observations: using all zeros as depth placeholder")
            observations[k] = torch.zeros([256, 256, 1])

        print("WARN: Using placeholders as the instruction, should be fixed later")
        instr = {'text': 'Walk forward down the hall past the table on the left. Continue going forward to you reach the open doorway to the left. Turn left and walk forward, stop in front of the doorway to the bathroom. Turn right and enter that hallway stop and wait in front of the sink on your right. ', 'tokens': [101, 3328, 2830, 2091, 1996, 2534, 2627, 1996, 2795, 2006, 1996, 2187, 1012, 3613, 2183, 2830, 2000, 2017, 3362, 1996, 2330, 7086, 2000, 1996, 2187, 1012, 2735, 2187, 1998, 3328, 2830, 1010, 2644, 1999, 2392, 1997, 1996, 7086, 2000, 1996, 5723, 1012, 2735, 2157, 1998, 4607, 2008, 6797, 2644, 1998, 3524, 1999, 2392, 1997, 1996, 7752, 2006, 2115, 2157, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'trajectory_id': 1660}
        observations['instruction'] = instr
        return observations
    
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
            ret = [self._get_observations()]
            return ret
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


            #debug_log(f"type(ret[0][0]), {type(ret[0][0])}") #h-Observations
            #debug_log(f"ret[0][0].keys(), {ret[0][0].keys()}") #h-Observations
            #dict_keys(['rgb', 'depth', 'rgb_30', 'rgb_60', 'rgb_90', 'rgb_120', 'rgb_150', 'rgb_180', 'rgb_210', 'rgb_240', 'rgb_270', 'rgb_300', 'rgb_330', 'depth_30', 'depth_60', 'depth_90', 'depth_120', 'depth_150', 'depth_180', 'depth_210', 'depth_240', 'depth_270', 'depth_300', 'depth_330', 'instruction']) 
            #for k,v in ret[0][0].items():
                #print(f"step: {k}, {type(v)}")
                #print(v if type(v) is dict else v.shape)
            #raise
            #debug_log(f"type(ret[0][1]), {type(ret[0][1])}") #float
            #debug_log(f"type(ret[0][2]), {type(ret[0][2])}") #bool, Task done?
            #debug_log(f"type(ret[0][3]), {type(ret[0][3])}") #h-Metrics
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
            return self.episodes
        else:
            raise NotImplementedError
        
        #current

    def close(self):
        """
        Closes the environment to free up resources.
        """
        self.envs.close()
