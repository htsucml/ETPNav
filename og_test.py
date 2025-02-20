import torch as th
import numpy as np
import cv2
import os
import yaml
import math
import imageio
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.tasks import GraspTask
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)
from omnigibson.action_primitives.action_primitive_set_base import (
    ActionPrimitiveError,
    ActionPrimitiveErrorGroup,
)
import transforms3d.euler as euler
import time


gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
gm.DEFAULT_VIEWER_WIDTH = 224
gm.DEFAULT_VIEWER_HEIGHT = 224

def get_config(scene_name, obj_config): #config preprossing
    cfg = {
        "render":{
            "viewer_width": 224,
            "viewer_height": 224,
        },
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

def execute_controller(ctrl_gen, env): #controller for grasping task, executing the plan stepwisely
    for action in ctrl_gen:
        env.step(action)


def pause_for_interaction(sim, print_position=False): #Pause function that allows keyboard interaction
    for i in range(10000):
        sim.step()
        position,orientation =  sim.viewer_camera.get_position_orientation()
        if print_position:
            print(position)

def turn_right(robot, delta=0.03):
    #rotating the robot. Also the built-in turn_left/right is broken.
    from omnigibson.utils.transform_utils import euler2quat, quat2mat, quat_multiply
    import torch
    quat = robot.get_position_orientation()[1]
    #inp = torch.tensor([-delta, 0, 0], dtype=torch.float32)
    inp = torch.tensor([0, 0, -delta], dtype=torch.float32)
    #quat = quat_multiply((euler2quat(-delta, 0, 0)), quat)
    quat = quat_multiply((euler2quat(inp)), quat)
    robot.set_position_orientation(orientation=quat)

def save_video(video_cache, save_path):
    video_writer = imageio.get_writer(save_path, fps=30)
    for rgb in video_cache:
        # print(type(rgb), rgb.shape) 
        if not isinstance(rgb, np.ndarray):
            rgb = np.array(rgb)
        video_writer.append_data(rgb)
    video_writer.close()



def main():
    #### Setup the config and initialize the env, ####  
    scene_name = 'Rs_int'
    obj_config = {'pos': [-0.01513892412185669, 1.4189201593399048, 1.0810081958770752], 'ori': [0.5094113349914551, -0.07470039278268814, 0.2355378270149231, -0.824282705783844], 'init_info': {'class_module': 'omnigibson.objects.dataset_object', 'class_name': 'DatasetObject', 'args': {'name': 'wooden_spoon_83', 'category': 'wooden_spoon', 'model': 'aeubta', 'prim_type': 0, 'uuid': 36377017, 'scale': [1.0, 1.0, 1.0], 'in_rooms': ['kitchen_0', 'living_room_0']}}}
    obj_config['category'] = obj_config['init_info']['args']['category']
    obj_config['name'] = obj_config['init_info']['args']['name']
    obj_config['type'] = "DatasetObject"
    cfg = get_config(scene_name, obj_config)       
    del cfg['task'] #Bug: The task triggers "Resetting error:  Could not infer dtype of JointPrim", don't know why
    
    #### Position the robot ####
    robot_position = [-1,2.5]
    #robot_position = [0., 0., 0.]
    cfg['robots'][0]['position'] = [0,0,0]
    cfg['robots'][0]['position'] = [robot_position[0], robot_position[1], cfg['robots'][0]['position'][2]]
    
    #### Setup the env and variables
    print(cfg)
    env = og.Environment(configs=cfg)
    scene = env.scene
    robot = env.robots[0]
    sim = og.sim
    standability_map = env.scene.trav_map


    #### Camera test ####

    from omnigibson.sensors import VisionSensor
    from omnigibson.utils.transform_utils import euler2quat, quat_multiply
    import torch
    viewer_camera = og.sim.viewer_camera
    #viewer_camera.viewer_width = 224
    #viewer_camera.viewer_height = 224
    print(f"sim viewer resolution: {sim.viewer_width}x{sim.viewer_height}")
    #print(f"viewer resolution: {viewer_camera}x{sim.viewer_height}")


    camera_fov = 90  # Field of View similar to MP3D/HM3D
    image_size = (224, 224)  # Target image size
    # Yaw angles for 30-degree increments

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
    output_path = "test_og_camera_pano_3.jpg" 
    cv2.imwrite(output_path, cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR))
    ''' #movement test
    for robot_position in [[0.0,0.0]]:
        new_position = robot_position + [robot.get_position()[2]]
        robot.set_position(new_position)
        sim.step()
        pano_images = get_pseudo_pano(robot, viewer_camera)
        stitched_image = cv2.hconcat(pano_images) 
        output_path = f"test_og_camera_pano_3_{robot.get_position()}.jpg"  # Set your desired filename
        #cv2.imwrite(output_path, cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(output_path, stitched_image)
    '''
    raise

    ####</camera test

    

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    controller = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
    #cabinet = scene.object_registry("name", "bottom_cabinet_slgzfc_0")
    print(robot.get_position())


   
    
    # Grasp apple
    print("Executing controller")
    sim.step()  #Workaround: "Resetting" the robot arm
    
 
    cam = og.sim.viewer_camera
    rgb_obs = np.array(cam.get_obs()[0]['rgb'])
    
    #Workaround: Manually rotating the robot to make it facing the object
    for _ in range(20): 
        time.sleep(0.2)
        turn_right(robot, delta=np.pi/36)
        sim.step()
        if _%1==0:
            print(f"turning right: {_+1}")
            print(robot.get_position_orientation())

    obj = scene.object_registry("name", "cologne")
    #debug primitive actions
    pos_on_obj = controller._sample_position_on_aabb_side(obj)
    pose_on_obj = [pos_on_obj, th.tensor([0, 0, 0, 1])]
    print(f"pose_on_obj: {pose_on_obj}")

    obj_rooms = (
        obj.in_rooms
        if obj.in_rooms
        else [env.scene._seg_map.get_room_instance_by_point(pose_on_obj[0][:2])]
    )
    print(f"obj_rooms: {obj_rooms}")

    distance_lo, distance_hi = 0.0, 5.0
    distance = (th.rand(1) * (distance_hi - distance_lo) + distance_lo).item()
    yaw_lo, yaw_hi = -math.pi, math.pi
    yaw = th.rand(1) * (yaw_hi - yaw_lo) + yaw_lo
    avg_arm_workspace_range = th.mean(controller.robot.arm_workspace_range[controller.arm])
    pose_2d = th.cat(
        [
            pose_on_obj[0][0] + distance * th.cos(yaw),
            pose_on_obj[0][1] + distance * th.sin(yaw),
            yaw + math.pi - avg_arm_workspace_range,
        ]
    )
    print(f"pose_2d: {pose_2d}")
    room_instance = controller.env.scene._seg_map.get_room_instance_by_point(pose_2d[:2]) 
    print(f"room instance:{room_instance}")
    #room_instance not in obj_rooms

    #execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, obj), env)
    success = True
    try:
        ctrl_gen = controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, obj)
        print(f"ctrl_gen: {ctrl_gen}")
        video_cache = []
        step_count = 0
        for action in ctrl_gen:
            #print("execute action in ctrl gen")
            cam = og.sim.viewer_camera
            rgb_obs = np.array(cam.get_obs()[0]['rgb'])
            video_cache.append(rgb_obs)
            env.step(action)
    except ActionPrimitiveErrorGroup:
        success = False
        save_path = f'./grasp_videos_0/{robot_position[0]}_{robot_position[1]}_failed.mp4'
        save_video(video_cache, save_path)
    


    
    if success:
        save_path = f'./grasp_videos_0/{robot_position[0]}_{robot_position[1]}_success.mp4'
        print("Finished executing grasp")
        save_video(video_cache, save_path)
    
    
    


if __name__ == "__main__":
    main()