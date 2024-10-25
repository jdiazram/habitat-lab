#based on https://gist.github.com/matsuren/af4783a14b4de6ea4bd52a5e47339ebc
import habitat
from habitat.config import read_write
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
import math
import magnum as mn  
import habitat_sim  
from omegaconf import OmegaConf  
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import batch_obs
import numpy as np  

FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
FINISH = "f"

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def example():
    config = habitat.get_config("benchmark/nav/pointnav/my_pointnav_habitat_test.yaml")
    print(f"config:{config}")

    #
    with read_write(config):
        CAMERA_NUM = 6
        orient = [
            [0, math.pi, 0],            # Back
            [-math.pi / 2, 0, 0],       # Down
            [0, 0, 0],                  # Front
            [0, math.pi / 2, 0],        # Right
            [0, 3 * math.pi / 2, 0],    # Left
            [math.pi / 2, 0, 0],        # Up
        ]

        sensor_uuids = []

        # clean sensors
        sim_sensors = config.habitat.simulator.agents.main_agent.sim_sensors
        sim_sensors.clear()

        # setup cameras
        for camera_id in range(CAMERA_NUM):
            camera_template = f"RGB_{camera_id}"
            print(f"camera_template:{camera_template}")

            # new configuration sensor
            camera_spec = habitat_sim.CameraSensorSpec()
            camera_spec.uuid = camera_template.lower()
            camera_spec.sensor_type = habitat_sim.SensorType.COLOR
            camera_spec.resolution = [512, 512]  # Aumentar resolución de las cámaras
            camera_spec.position = mn.Vector3(0.0, 1.25, 0.0)
            camera_spec.orientation = mn.Vector3(orient[camera_id])
            camera_spec.hfov = 90.0
            camera_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

            # dict
            camera_config = OmegaConf.create()
            camera_config.type = 'HabitatSimRGBSensor'
            camera_config.uuid = camera_spec.uuid
            camera_config.height = camera_spec.resolution[0]
            camera_config.width = camera_spec.resolution[1]
            camera_config.position = [
                camera_spec.position.x,
                camera_spec.position.y,
                camera_spec.position.z
            ]
            camera_config.orientation = [
                camera_spec.orientation.x,
                camera_spec.orientation.y,
                camera_spec.orientation.z
            ]
            camera_config.hfov = float(camera_spec.hfov)
            camera_config.sensor_subtype = 'PINHOLE'
            camera_config.noise_model = 'None'
            camera_config.noise_model_kwargs = {}

            sensor_uuid = camera_spec.uuid
            sensor_uuids.append(sensor_uuid)
            sim_sensors[sensor_uuid] = camera_config

    # CubeMap2Equirect
    cube2equirec = baseline_registry.get_obs_transformer("CubeMap2Equirect")
    cube2equirec = cube2equirec(sensor_uuids, (512, 1024))  # size

    # env
    env = habitat.Env(config=config)
    print("Environment creation successful")

    observations = env.reset()

    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))

    # 
    batch = batch_obs([observations])
    trans_observations = cube2equirec(batch)
    equirect_image = trans_observations['rgb_0'].numpy().squeeze()

    # scale 
    scale_factor = 1.0
    equirect_image_resized = cv2.resize(
        equirect_image,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_LINEAR
    )

    # 
    cv2.imshow("Equirectangular Image", transform_rgb_bgr(equirect_image_resized))

    count_steps = 0
    while not env.episode_over:
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.move_forward
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.turn_left
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.turn_right
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.stop
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1

        print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            observations["pointgoal_with_gps_compass"][0],
            observations["pointgoal_with_gps_compass"][1]))

        # 
        batch = batch_obs([observations])
        trans_observations = cube2equirec(batch)
        equirect_image = trans_observations['rgb_0'].numpy().squeeze()

        # 
        equirect_image_resized = cv2.resize(
            equirect_image,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LINEAR
        )

        # 
        cv2.imshow("Equirectangular Image", transform_rgb_bgr(equirect_image_resized))

    print("Episode finished after {} steps.".format(count_steps))

    if (
        action == HabitatSimActions.stop
        and observations["pointgoal_with_gps_compass"][0] < 0.2
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    example()
