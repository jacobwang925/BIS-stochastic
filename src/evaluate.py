import pickle
import os, sys, glob
import numpy as np
from models import *
from agents import *
import copy
# from utils.World import World
import time

from tqdm import tqdm

def evaluate(model, algorithm, graphics = False, robot = None, save_postfix=None, passive = True):
    """This function evaluate an algorithm's performance on a given model. To avoid repeated test, the algorithm will examine whether the results already exist. If they do, it will return the results directly. Which also means if you updated some algorithm, you need to delete previous results manually to see the changes.

    Args:
        model (str): the name of the model to be tested
        algorithm (str): the name of the algorithm to be tested
        graphics (bool): whether to use graphics to show the evaluation process on live
        robot (KinematicModel): you can pass in an initialized agent with a given setting, or the function will use a default one. This is useful when you grid search the parameters.
        save_postfix (str): a string to specify the name of the results
        passive(bool): whether the human model is passive to the robot's behavior. If is ture, human model will not react to the robot.

    Returns:
        total_score (dict): A dict contains the algorithm's average score on different aspects, include safety, efficiency, collision count, and nearest distance.

    """


    
    if save_postfix is None:
        save_postfix = 'best'
    
    save_dir = os.path.join('eval_results', model, algorithm, save_postfix)
    if not passive:
        save_dir = os.path.join('interactive_eval_results', model, algorithm, save_postfix)
    
    # Avoid repetition, which also means if you updated some algorithm, you need to delete previous results manually to see the changes.

    if robot is None:
        robot = eval(model + '(' + algorithm + '(), dT)')
    
        
    if glob.glob1(save_dir, 'total_score'):
        f = open(os.path.join(save_dir, 'total_score'), 'rb')
        total_score = pickle.load(f)
        # print('total_score')
        # print(total_score)
        return total_score
    
    dT = 0.02
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data_dir = 'simulate_data'
    names = glob.glob1(data_dir, 'data_*')
    
    total_score = dict()

    # start_time = time.time()
    # n = len(names[1:4])
    # print(names[1:3])
    n = len(names)
    # print(n)
    # for name in names:
    # for name in names[1:5]:
    for name in names[5:15]:
        # print(name)
        f = open(os.path.join(data_dir, name), 'rb')

        record = pickle.load(f)
        record.robot_moves = np.matrix(np.zeros((np.shape(robot.x)[0], record.tot)))
        record.cnt = 0

        if passive:
            if robot.is_2D:
                human = HumanBall2D(MobileAgent(), dT)
            else:
                human = HumanBall3D(MobileAgent(), dT)
        else:
            if robot.is_2D:
                human = InteractiveHumanBall2D(SafeSet(d_min=1, k_v=1), dT)
            else:
                human = InteractiveHumanBall3D(SafeSet(d_min=1, k_v=1), dT)


        #Make sure all the algorithms have same goal sequence
        human.reset(record.dT, record.human_goals)
        robot.reset(record.dT, record.robot_goals)
        
        score = dict()

        record.model = model
        record.algorithm = algorithm

        # start_time = time.time()
        for t in tqdm(range(record.tot)):
        # for t in range(300):
            human.update(robot)
            if passive:
                human.move(*record.human_moves[:,t])
            else:
                human.move()
                if np.shape(human.x)[0] == 4:
                    record.human_moves[:, t] = np.vstack([human.x[[0,1]], 0, human.x[[2,3]], 0])
                else:
                    record.human_moves[:, t] = human.x

            if algorithm == 'StoSafe':
                ## todo
                x0 = None # get state of the robot
                # F = evaluate_safety(model, algorithm, robot, human, dx = 0.1, H = 30)
                start_time = time.time()
                # F = evaluate_safety(model, algorithm, robot, human, dx=0.1, H=1)  # around 1 s
                F = evaluate_safety(model, algorithm, robot, human, dx=0.1, H=20)  # around 30 s
                # print('F:')
                # print(F)
                # dF = None
                robot.update(human, Sto=True, MC=False, F=F)
                print("--- %s seconds ---" % (time.time() - start_time))
            else:
                robot.update(human)
            robot.move()
            record.robot_moves[:, t] = robot.x

            # print(record.tot)
            # print(t)
        # print("--- %s seconds ---" % (time.time() - start_time))
        # if graphics:
            
        #     try:
        #         w = World(record.dT, human, robot, record)
        #         base.run()
        #     except SystemExit as e:
        #         pass

        save_data(save_dir, name.replace('data', 'result'), record)
        for k in robot.score.keys():
            if k not in total_score:
                total_score[k] = robot.score[k] / n
            else:
                total_score[k] = total_score[k] + robot.score[k] / n

        if not passive:
            total_score['efficiency'] += human.score['efficiency'] / n

        # print(name)

        # print('score[efficiency]')
        # print(robot.score['efficiency'])
        # print('score[collision_cnt]')
        # print(robot.score['collision_cnt'])
    # print('total_score')
    # print(total_score)
    # print("--- %s seconds ---" % (time.time() - start_time))
    print('total_score[efficiency]')
    print(total_score['efficiency'])
    print('total_score[collision_cnt]')
    print(total_score['collision_cnt'])
    save_data(save_dir, 'total_score', total_score)
    return total_score


def evaluate_safety(model, algorithm, robot, human, dx = 0.1, H = 10, graphics=False, passive=True):
    """This function evaluate an algorithm's performance on a given model. To avoid repeated test, the algorithm will examine whether the results already exist. If they do, it will return the results directly. Which also means if you updated some algorithm, you need to delete previous results manually to see the changes.

    Args:
        model (str): the name of the model to be tested
        algorithm (str): the name of the algorithm to be tested
        graphics (bool): whether to use graphics to show the evaluation process on live
        robot (KinematicModel): you can pass in an initialized agent with a given setting, or the function will use a default one. This is useful when you grid search the parameters.
        save_postfix (str): a string to specify the name of the results
        passive(bool): whether the human model is passive to the robot's behavior. If is ture, human model will not react to the robot.

    Returns:
        total_score (dict): A dict contains the algorithm's average score on different aspects, include safety, efficiency, collision count, and nearest distance.

    """

    # Avoid repetition, which also means if you updated some algorithm, you need to delete previous results manually to see the changes.

    dT = 0.02

    ndim = 4
    human_t = human

    # episodes = 10 # Monte Carlo episode number
    episodes = 20  # Monte Carlo episode number

    safety = np.zeros([episodes, 3, ndim])

    count = robot.score['collision_cnt']

    for eps in range(episodes):
        for n in range(ndim):

            # x = x0 - dx
            ## todo

            robot_t = copy.deepcopy(robot)

            x0 = robot_t.x
            # robot_t ?? <- x0 + dx
            robot_t.x[n] -= dx # maybe
            # robot_t.reset(record.dT, record.robot_goals) # reset initial state of robot
            human_t = copy.deepcopy(human) # temp human for looping


            for t in range(H):
                human_t.update(robot_t)
                if passive:
                    human_t.move()
                else:
                    pass

                robot_t.update(human_t)
                robot_t.move()

            if robot_t.score['collision_cnt'] - count > 0:
                safety[eps, 0, n] = 0
            else:
                safety[eps, 0, n] = 1

            # x = x0
            ## todo
            robot_t = copy.deepcopy(robot)
            # robot_t.reset(record.dT, record.robot_goals)  # reset initial state of robot
            human_t = copy.deepcopy(human)  # temp human for looping

            for t in range(H):
                human_t.update(robot_t)
                if passive:
                    human_t.move()
                else:
                    pass

                robot_t.update(human_t)
                robot_t.move()

            if robot_t.score['collision_cnt'] - count > 0:
                safety[eps, 1, n] = 0
            else:
                safety[eps, 1, n] = 1

            # x = x0 + dx
            robot_t = copy.deepcopy(robot)
            robot_t.x[n] += dx  # maybe
            # robot_t.reset(record.dT, record.robot_goals)  # reset initial state of robot
            human_t = copy.deepcopy(human)  # temp human for looping

            # start_time = time.time()
            for t in range(H):  # 0.25s
                human_t.update(robot_t)
                if passive:
                    human_t.move()
                else:
                    pass

                robot_t.update(human_t)
                robot_t.move()
            # print("--- %s seconds ---" % (time.time() - start_time))

            if robot_t.score['collision_cnt'] - count > 0:
                safety[eps, 2, n] = 0
            else:
                safety[eps, 2, n] = 1

    F = safety.mean(axis=0)

    return F

def save_data(folder, name, record):
    """
    This function saves the results.

    Args:
        folder: folder path
        name: file name
        record: evaluation result
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    f = open(os.path.join(folder, name), 'wb')
    print(os.path.join(folder, name))
    pickle.dump(record, f)
    
if __name__ == "__main__":
    graphics = False
    evaluate(sys.argv[1], sys.argv[2], graphics=graphics)