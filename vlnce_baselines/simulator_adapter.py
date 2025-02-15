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
            from habitat_baselines.common.env_utils import construct_envs, get_env_class
            self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))

            self.num_envs = self.envs.num_envs
            self.number_of_episodes = self.envs.number_of_episodes
            self.observation_spaces = self.envs.observation_spaces
            self.action_spaces = self.envs.observation_spaces

        elif self.simulator_type == "omnigibson":
            import omnigibson as og
            self.envs = og.Environment(self.config)  # Placeholder for actual OG setup
        else:
            raise ValueError(f"Unsupported simulator type: {self.simulator_type}")

    def reset(self):
        """
        Resets the environment and returns the initial observation.
        """
        if self.simulator_type == "habitat":
            return self.envs.reset()
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
            Tuple (observations, rewards, dones, infos)
        """
        if self.simulator_type == "habitat":
            return self.envs.step(actions)
        elif self.simulator_type == "omnigibson":
            return self.envs.step(actions)
        else:
            raise NotImplementedError

    def pause_at(self, index):
        """
        Pauses an environment at the given index.
        Useful for managing multiple environments in parallel.

        Args:
            index (int): Index of the environment to pause.
        """
        if self.simulator_type == "habitat":
            self.envs.pause_at(index)
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
            self.envs.resume_all()
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

    def close(self):
        """
        Closes the environment to free up resources.
        """
        self.envs.close()
