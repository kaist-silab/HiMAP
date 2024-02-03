import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List
import numpy as np
from collections import deque
import pickle
import config

import sys; sys.path.append('./')
from src.models.vggnet import VGG_Net

KEEP_PROB1             = 1 # was 0.5
KEEP_PROB2             = 1 # was 0.7
RNN_SIZE               = 512
GOAL_REPR_SIZE         = 12
SOFTMAX_TEMPERATURE    = config.SOFTMAX_TEMPERATURE
HISTORY_SIZE           = config.HISTORY_SIZE
HISTORY_THRESHOLD      = 0 
ngpu                   = config.ngpu

TEST_DATA_PATH = config.TEST_DATA_PATH

MAP_SIZE = config.TEST_MAP_SIZE
MAX_TIMESTEP = config.TEST_MAX_TIMESTEP
action_list = np.array([[1, 0],[0, 1],[-1, 0],[0, -1],[0, 0]], dtype=int)
model_path = config.model_path
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


model = VGG_Net(a_size=5, rnn_size=RNN_SIZE, goal_repr_size=GOAL_REPR_SIZE, keep_prob1=KEEP_PROB1, keep_prob2=KEEP_PROB2, softmax_temperature=SOFTMAX_TEMPERATURE)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
print('Trained model loaded!')
print('-----------------------------------')


def load_pickled_data(file_path, scen):
    
    with open(file_path, 'rb') as f:
        test_scen = pickle.load(f)
    
    map_array=test_scen[scen][0]
    map_array= np.where(map_array == 1, -1, map_array)
    map_array=map_array.astype(int)
    scen_data=[]
    for agent,goal in zip(test_scen[scen][1],test_scen[scen][2]):
        pair1=(int(agent[0]),int(agent[1]))
        pair2=(int(goal[0]),int(goal[1]))
        scen_data.append((pair1,pair2))
    num_agents=len(scen_data)  
    return map_array,scen_data,num_agents  

def add_S_G(scen_data,map_array):
    '''add start and goal locations to the map_array'''
    # Initialize state and goal arrays
    state_array = map_array.copy() 
    goal_array = np.zeros((map_array.shape[0], map_array.shape[1]), dtype=int)

    for agent_id,(startloc,goalloc) in enumerate(scen_data,1):
        goal_array[goalloc] = agent_id  # Set agent's goal in the goal array
        state_array[startloc] = agent_id   # Set agent's position in the state array
           
    return state_array, goal_array

def scan_A_G(num_agents,state_array,goals_array,map_array):
    '''
    scan the map and return the locations of the agents and goals
    output: two lists of tuples, each tuple is the location of an agent or a goal
    '''
    goals_locations = [(-1,-1) for _ in range(num_agents)]
    agents_locations = [(-1,-1) for _ in range(num_agents)]
    for i in range(map_array.shape[0]):
        for j in range(map_array.shape[1]):
         
            if(goals_array[i,j]>0):
                goals_locations[goals_array[i,j]-1] = (i,j) #minus 1 because agent one is 0-th element in the array
            if(state_array[i,j]>0):
                agents_locations[state_array[i,j]-1] = (i,j)
    assert((-1,-1) not in goals_locations)
    assert((-1,-1) not in agents_locations)
    return  agents_locations,goals_locations

def scan_A(num_agents,state_array,map_array):
    '''
    scan the map and return the locations of the agents from the second scanning
    '''
    agents_locations = [(-1,-1) for _ in range(num_agents)]
    for i in range(map_array.shape[0]):
        for j in range(map_array.shape[1]):
            if(state_array[i,j]>0):
                agents_locations[state_array[i,j]-1] = (i,j)

    return agents_locations

def observe(agent_id,
            agents_locations,
            goals_locations,
            state_array,
            goal_array,
            map_array,
            observation_size=11):
        
        '''
        observe the environment from the perspective of agent_id
        FOV is a square of size observation_size*observation_size
        INPUTS:
            agent_id: the id of the agent observing the environment, must be >0
            observation_size: the size of the observation square
            map_array: the ORIGINAL map of the environment
            agents_locations: the locations of the agents
            goals_locations: the locations of the goals
            state_array: the state of the environment
            goal_array: the goals map 
        '''

        assert(agent_id>0)
     

        top_left=(agents_locations[agent_id-1][0]-observation_size//2,agents_locations[agent_id-1][1]-observation_size//2)
        bottom_right=(top_left[0]+observation_size,top_left[1]+observation_size)        
        obs_shape=(observation_size,observation_size)
        goal_map             = np.zeros(obs_shape)
        poss_map             = np.zeros(obs_shape)
        goals_map            = np.zeros(obs_shape)
        obs_map              = np.zeros(obs_shape)
        visible_agents=[]
        for i in range(top_left[0],top_left[0]+observation_size):
            for j in range(top_left[1],top_left[1]+observation_size):
                if i>=map_array.shape[0] or i<0 or j>=map_array.shape[1] or j<0:
                    #out of bounds, just treat as an obstacle
                    obs_map[i-top_left[0],j-top_left[1]]=1
                    continue
                if state_array[i,j]==-1:
                    #obstacles
                    obs_map[i-top_left[0],j-top_left[1]]=1
                if state_array[i,j]==agent_id:
                    #agent's position
                    poss_map[i-top_left[0],j-top_left[1]]=1
                if goal_array[i,j]==agent_id:
                    #agent's goal
                    goal_map[i-top_left[0],j-top_left[1]]=1
                if state_array[i,j]>0 and state_array[i,j]!=agent_id:
                    #other agents' positions
                    visible_agents.append(state_array[i,j])
                    poss_map[i-top_left[0],j-top_left[1]]=1
        
        # Get visible agents - the values from state_array where the other_agents_mask is True
        for agent in visible_agents:
            x, y = goals_locations[agent-1]
            min_node = (max(top_left[0], min(top_left[0] + observation_size - 1, x)),
                        max(top_left[1], min(top_left[1] + observation_size - 1, y)))
            goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        dx=goals_locations[agent_id-1][0]-agents_locations[agent_id-1][0]
        dy=goals_locations[agent_id-1][1]-agents_locations[agent_id-1][1]
        mag=(dx**2+dy**2)**.5
        if mag!=0:
            dx=dx/mag
            dy=dy/mag

        return (np.concatenate((poss_map[np.newaxis, :],goal_map[np.newaxis, :],goals_map[np.newaxis, :],obs_map[np.newaxis, :]),axis=0),np.array([dx,dy,mag]))

def NextSamplingAction(num_agents,
               agents_locations,
               goals_locations,
               state_array,
               goal_array,
               map_array,
               agent_histories
               ):
    ''' 
    Get next actions for multiple agents at the same time and handle collisions.
    '''
    def check_visited_more_than_twice(agent_id, x, y):
        return sum([1 for pos in agent_histories[agent_id] if pos == (x,y)]) > HISTORY_THRESHOLD
    
    observations = []
    unit_vectors = []
    curr_positions = []

    # Gather observations for all agents
    for agent_id in range(1, num_agents + 1):
        
        curr_x, curr_y = agents_locations[agent_id - 1]
        obs = observe(agent_id, agents_locations, goals_locations, state_array, goal_array, map_array)

        observation, unit_vector_and_magnitude = obs
        observations.append(torch.tensor(observation, dtype=torch.float32).to(device))
        unit_vectors.append(torch.tensor(unit_vector_and_magnitude, dtype=torch.float32).to(device))
        curr_positions.append((curr_x, curr_y))

    # Convert list of tensors to a single batch tensor
    observations_tensor = torch.stack(observations).to(device)
    unit_vectors_tensor = torch.stack(unit_vectors).to(device)
    
    with torch.no_grad():
        # Get policy distribution for the batch
        # This parallelizes the forward pass for all agents
        policies = model(observations_tensor, unit_vectors_tensor)
        action_probs = policies.cpu()
    
    def is_unavailable_action(curr_x, curr_y, action):
        next_x = curr_x + action_list[action][0]
        next_y = curr_y + action_list[action][1]
        return (next_x < 0 or next_x >= map_array.shape[0] or next_y < 0 or next_y >= map_array.shape[1]) or \
            (map_array[next_x, next_y] == -1) or \
            check_visited_more_than_twice(agent_id-1, next_x, next_y)  
            
    # Make probability 0 for unavailable actions
    for agent_id in range(1, num_agents + 1):
        curr_x, curr_y = curr_positions[agent_id - 1]
        for action in range(5):
            if is_unavailable_action(curr_x, curr_y, action):
                action_probs[agent_id - 1][action] = 0
        # if all actions are unavailable, then set the probability of staying to 1
        if torch.sum(action_probs[agent_id - 1]) == 0:
            action_probs[agent_id - 1][4] = 1
                
    # Sample actions for all agents
    actions = torch.multinomial(action_probs, 1).squeeze().tolist()

    # Set probability to 1 for agents that are already at the goal
    for idx, pos in enumerate(curr_positions):
        if pos == (-1, -1):
            actions[idx] = 4 # this forces the agent to stay at the goal 
    
    # NOTE
    # We are not handling collisions here, since they can be handled in the step function
    return actions

def step(actions: List[int],num_agents,agents_pos,goals_pos,map_array):
        '''
        my actions:
            list of indices
            0: down, 1: right, 2: up, 3: left, 4: stay
        '''

        assert len(actions) == num_agents, 'only {} actions as input while {} agents in environment'.format(len(actions), num_agents)
        assert all([action_idx<=4 and action_idx>=0 for action_idx in actions]), 'action index out of range'

        agents_pos = np.array(agents_pos)
        checking_list = [i for i in range(num_agents)]
        next_pos = np.copy(agents_pos)
        map_size = map_array.shape

        # remove unmoving agent id
        for agent_id in checking_list.copy():
            if actions[agent_id] == 4: #change to 4 from 0
                # unmoving
                checking_list.remove(agent_id)
            else:
                # move
                next_pos[agent_id] += action_list[actions[agent_id]]

        # first round check, these two conflicts have the heightest priority
        for agent_id in checking_list.copy():

            if np.any(next_pos[agent_id]<0) or np.any(next_pos[agent_id]>=map_size[0]):
                # agent out of map range
                next_pos[agent_id] = agents_pos[agent_id]
                checking_list.remove(agent_id)

            elif map_array[tuple(next_pos[agent_id])] == -1:
                # collide obstacle
                next_pos[agent_id] = agents_pos[agent_id]
                checking_list.remove(agent_id)

        # second round check, agent swapping conflict
        no_conflict = False
        while not no_conflict:

            no_conflict = True
            for agent_id in checking_list:

                target_agent_id = np.where(np.all(next_pos[agent_id]==agents_pos, axis=1))[0]

                if target_agent_id.size > 0:

                    target_agent_id = target_agent_id.item()
                    assert target_agent_id != agent_id, 'logic bug, the agent did not move, should not be in checking list'

                    if np.array_equal(next_pos[target_agent_id], agents_pos[agent_id]):
                        assert target_agent_id in checking_list, 'target_agent_id should be in checking list'

                        next_pos[agent_id] = agents_pos[agent_id]

                        next_pos[target_agent_id] = agents_pos[target_agent_id]

                        checking_list.remove(agent_id)
                        checking_list.remove(target_agent_id)

                        no_conflict = False
                        break

        # third round check, agent collision conflict
        no_conflict = False
        while not no_conflict:
            no_conflict = True
            for agent_id in checking_list:

                collide_agent_id = np.where(np.all(next_pos==next_pos[agent_id], axis=1))[0].tolist()
                if len(collide_agent_id) > 1:
                    # collide agent
                    
                    # if all agents in collide agent are in checking list
                    all_in_checking = True
                    for id in collide_agent_id.copy():
                        if id not in checking_list:
                            all_in_checking = False
                            collide_agent_id.remove(id)

                    if all_in_checking: 

                        collide_agent_pos = next_pos[collide_agent_id].tolist()
                        for pos, id in zip(collide_agent_pos, collide_agent_id):  #this is possible because in python, list is passed by reference
                            pos.append(id)  #changes 'collide_agent_pos'
                        collide_agent_pos.sort(key=lambda x: x[0]*map_size[0]+x[1])

                        collide_agent_id.remove(collide_agent_pos[0][2])

                        # checking_list.remove(collide_agent_pos[0][2])

                    next_pos[collide_agent_id] = agents_pos[collide_agent_id]

                    for id in collide_agent_id:
                        checking_list.remove(id)

                    no_conflict = False
                    break
        

        agents_pos = np.copy(next_pos)
        
        for i in range(num_agents):
            if np.array_equal(agents_pos[i], goals_pos[i]):
                agents_pos[i] = (-1,-1)

        done = False
        # if np.unique(agents_pos, axis=0).shape[0] == 1:
        if np.all(np.array(agents_pos)==(-1,-1)):
            done = True

        # make sure no overlapping agents
        detect_overlap=set()
        for i in range(num_agents):
            if tuple(agents_pos[i]) == (-1,-1):
                continue
            if tuple(agents_pos[i]) in detect_overlap:
                raise RuntimeError('overlapping agents')
            detect_overlap.add(tuple(agents_pos[i]))
        
        # Write the next_pos of agents to the state_array and delete the previous position
        state_array = map_array.copy() 
        for agent_id in range(num_agents):
            if tuple(agents_pos[agent_id])!=(-1,-1):
                state_array[tuple(agents_pos[agent_id])] = agent_id+1
            else:
                state_array[tuple(agents_pos[agent_id])] = -1
        
        return  done,state_array

def Accuracy_on_Given_Dim_NumAgent(dim=40,agent_num=32,max_timestep=MAX_TIMESTEP):
    '''
    Test on all cases of map(size=dim) given the specific number of agents
    return: a list of success rate
    '''
    success=[]
    for scen in range(config.NUM_TEST_CASES):  
        map_array, scen_data, num_agents = load_pickled_data(TEST_DATA_PATH + f"{dim}length_{agent_num}agents_0.3density.pth", scen)
        state_array,goal_array=add_S_G(scen_data,map_array)
        agents_pos,goals_pos=scan_A_G(num_agents,state_array,goal_array,map_array)
        # Initialize agent histories
        agent_histories = [deque(maxlen=HISTORY_SIZE) for _ in range(num_agents)]
        done=False
        for i in range(max_timestep):
            agents_pos=scan_A(num_agents,state_array,map_array)
            if done:
                print(f'All agents reached the goals at {i}th timestep.')
                success_rate=1.0
                break
            elif i==max_timestep-1:
                
                count = sum(1 for item in agents_pos if item == (-1, -1))
                print(f'Not all agents reached the goals, {count}/{num_agents} agents reached the goals.')
                success_rate=count/num_agents
            actions=NextSamplingAction(num_agents,agents_pos,goals_pos,state_array,goal_array,map_array, agent_histories)
            for idx, pos in enumerate(agents_pos):
                agent_histories[idx].append(pos)    
                
            done,state_array=step(actions,num_agents,agents_pos,goals_pos,map_array)
        success.append(success_rate)
    return success

if __name__ == "__main__":
    success_dict={}
    for agents in config.NUM_TEST_AGENTS: 
        print(f"Map Size: {MAP_SIZE}, Agent number: {agents}, Temperature: {SOFTMAX_TEMPERATURE}, History Size: {HISTORY_SIZE}")
        success_rate=Accuracy_on_Given_Dim_NumAgent(dim=MAP_SIZE,agent_num=agents,max_timestep=MAX_TIMESTEP)
        success_dict[agents]=success_rate
        print(f"Average success rate for {agents} agents: {np.mean(success_rate)}")
        print('-----------------------------------')

    #compute the average success rate for each agent number and save it to a dict
    average_success_dict={}
    for key in success_dict.keys():
        average_success_dict[key]=np.mean(success_dict[key])
    print('Average success rate for each agent number:')
    print(average_success_dict)