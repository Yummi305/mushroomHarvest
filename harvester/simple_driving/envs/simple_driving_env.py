import gym
import numpy as np
import math
import pybullet as p
from pybullet_utils import bullet_client as bc
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
import matplotlib.pyplot as plt
import time

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class SimpleDrivingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'fp_camera', 'tp_camera']}

    def __init__(self, isDiscrete=True, renders=False):
        if (isDiscrete):
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -.6], dtype=np.float32),
                high=np.array([1, .6], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-40, -40, -1, -1, -5, -5, -10, -10], dtype=np.float32),
            high=np.array([40, 40, 1, 1, 5, 5, 10, 10], dtype=np.float32))
        self.np_random, _ = gym.utils.seeding.np_random()

        self.mushrooms = []  # Initialize the mushrooms attribute as an empty list
        self.mushroom = None  # Initialize the mushroom object

        ## Insert code to randomise our mushroom and determine if it is edible or not.
        print("A random mushroom has been spawned. Robot will investigate.")
        
        
        # Initialize isEdible randomly at the beginning of the training loop
        #self.isEdible = bool(np.random.randint(2))  # Randomly set to True or False
        
        # Test for is edible
        self.isEdible = True
        
        # Test for is not edible
        # self.isEdible = False
        
        # Check edible status
        print("Is the random mushroom edible?: ", self.isEdible)

        if renders:
          self._p = bc.BulletClient(connection_mode=p.GUI)
        else:
          self._p = bc.BulletClient()

        self.reached_goal = False
        self._timeStep = 0.01
        self._actionRepeat = 50
        self._renders = renders
        self._isDiscrete = isDiscrete
        self.car = None
        self.goal_object = None
        self.goal = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.reset()
        self._envStepCounter = 0
    
        

    def get_reached_goal_status(self):
        return self.reached_goal

    def step(self, action):
        # Feed action to the car and get observation of car's state
        if (self._isDiscrete):
            fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
            throttle = fwd[action]
            steering_angle = steerings[action]
            action = [throttle, steering_angle]
        self.car.apply_action(action)
        for i in range(self._actionRepeat):
          self._p.stepSimulation()
          if self._renders:
            time.sleep(self._timeStep)

          carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
          goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
          car_ob = self.getExtendedObservation()

          if self._termination():
            self.done = True
            break
          self._envStepCounter += 1

        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((carpos[0] - goalpos[0]) ** 2 +
                                  (carpos[1] - goalpos[1]) ** 2))
        reward = -dist_to_goal
        self.prev_dist_to_goal = dist_to_goal

        # Calculate distance to mushroom
        min_dist_to_mushroom = float('inf')
        mushroom_pos, _ = self._p.getBasePositionAndOrientation(self.mushroom)
        dist_to_mushroom = math.sqrt((carpos[0] - mushroom_pos[0]) ** 2 + (carpos[1] - mushroom_pos[1]) ** 2)
        min_dist_to_mushroom = min(min_dist_to_mushroom, dist_to_mushroom)

        # Penalize if too close to mushroom
        if min_dist_to_mushroom < 1:
            if self.isEdible:
                print(f"Sticking close to mushroom")
                reward += 10
            else:
                print(f"Phew, close call!")
                reward -= 10

        # Mushroom collection if edible
        if self.isEdible:
            if min_dist_to_mushroom <= 0.5:
                print(f"Mushroom collected!")
                reward += 30
                self._p.changeVisualShape(self.mushroom, 0, rgbaColor=[0, 0, 0, 0])
                self._p.changeVisualShape(self.mushroom, 1, rgbaColor=[0, 0, 0, 0])
                
        # Done by reaching goal
        if dist_to_goal < 1.5 and not self.reached_goal:
            reward += 100
            print("REACHED GOAL")
            self.done = True
            self.reached_goal = True
        
        ob = car_ob
        return ob, reward, self.done, dict()



    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        # Reload the plane and car
        Plane(self._p)
        self.car = Car(self._p)
        self._envStepCounter = 0

        # Set the goal to a random target
        x = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
            self.np_random.uniform(-9, -5))
        y = (self.np_random.uniform(5, 9) if self.np_random.integers(2) else
            self.np_random.uniform(-9, -5))
        self.goal = (x, y)
        self.done = False
        self.reached_goal = False

        # Visual element of the goal
        self.goal_object = Goal(self._p, self.goal)

        # Calculate mushroom position between car and goal
        carpos, _ = self._p.getBasePositionAndOrientation(self.car.car)
        goalpos, _ = self._p.getBasePositionAndOrientation(self.goal_object.goal)
        mushroom_x = (carpos[0] + goalpos[0]) / 2
        mushroom_y = (carpos[1] + goalpos[1]) / 2
        mushroom_pos = [mushroom_x, mushroom_y, 0.5]

        # Load mushrooms
        if self.isEdible:
            self.mushroom = self._p.loadURDF(fileName="simple_driving/resources/simplemushroom.urdf", basePosition=mushroom_pos)
        else:
            self.mushroom = self._p.loadURDF(fileName="simple_driving/resources/poisonmushroom.urdf", basePosition=mushroom_pos)

        # Get observation to return
        carpos = self.car.get_observation()

        self.prev_dist_to_goal = math.sqrt(((carpos[0] - self.goal[0]) ** 2 + (carpos[1] - self.goal[1]) ** 2))
        car_ob = self.getExtendedObservation()
        return np.array(car_ob, dtype=np.float32)



    def render(self, mode='human'):
        if mode == "fp_camera":
            # Base information
            car_id = self.car.get_ids()
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                       nearVal=0.01, farVal=100)
            pos, ori = [list(l) for l in
                        self._p.getBasePositionAndOrientation(car_id)]
            pos[2] = 0.2

            # Rotate camera direction
            rot_mat = np.array(self._p.getMatrixFromQuaternion(ori)).reshape(3, 3)
            camera_vec = np.matmul(rot_mat, [1, 0, 0])
            up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
            view_matrix = self._p.computeViewMatrix(pos, pos + camera_vec, up_vec)

            # Display image
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame

        elif mode == "tp_camera":
            car_id = self.car.get_ids()
            base_pos, orn = self._p.getBasePositionAndOrientation(car_id)
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                    distance=20.0,
                                                                    yaw=40.0,
                                                                    pitch=-35,
                                                                    roll=0,
                                                                    upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                             aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
        else:
            return np.array([])

    def getExtendedObservation(self):
        carpos, carorn = self._p.getBasePositionAndOrientation(self.car.car)
        goalpos, goalorn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
        invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
        goalPosInCar, goalOrnInCar = self._p.multiplyTransforms(invCarPos, invCarOrn, goalpos, goalorn)

        # Calculate distance to closest mushroom
        min_dist_to_mushroom = float('inf')
        for mushroom in self.mushrooms:
            mushroom_pos, _ = self._p.getBasePositionAndOrientation(mushroom)
            dist_to_mushroom = math.sqrt((carpos[0] - mushroom_pos[0]) ** 2 + (carpos[1] - mushroom_pos[1]) ** 2)
            min_dist_to_mushroom = min(min_dist_to_mushroom, dist_to_mushroom)

        observation = [goalPosInCar[0], goalPosInCar[1], min_dist_to_mushroom]
        return observation

    def _termination(self):
        return self._envStepCounter > 2000 # Max steps.

    def close(self):
        self._p.disconnect()

# Create environment
env = SimpleDrivingEnv(isDiscrete=True, renders=False)
