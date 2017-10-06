import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 8
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 20
FRESH_TIME = 0.1

def build_q_table(n_states, actions):
	table = pd.DataFrame(
		np.zeros((n_states, len(actions))),#np.zeros((row, column), data type)
		columns = actions,
	)
	#print(table)
	return table
#build_q_table(N_STATES, ACTIONS)

def choose_action(state, q_table):
	state_actions = q_table.iloc[state, :]#iloc[nth row, mth column]
	if (np.random.uniform() > EPSILON ) or (state_actions.all() == 0):
		action_name = np.random.choice(ACTIONS)#generate a random sample from a given 1-D array
	else:
		action_name = state_actions.argmax()#return the index of the max value
	return action_name


def get_env_feedback(S, A):
	if A == 'right':
		if S == N_STATES - 2:#terminate
			S_ = 'terminal'
			R = 1
		else:
			S_ = S + 1
			R = 0
	else:#left
		if S == 0:#meet the wall
			S_ = S
		else:
			S_ = S - 1
		R = 0
	return S_, R

def update_env(S, episode, step_counter):
	env_list = ['-']*(N_STATES - 1) + ['#']
	if S == 'terminal':
		reaction = 'Episode %s: total steps = %s' % (episode+1, step_counter)
		print('\r{}'.format(reaction), end = '')
		time.sleep(1)
		print('\r                                ',end = '')
	else:
		env_list[S] = 'o'
		reaction = ''.join(env_list)
		print('\r{}'.format(reaction), end = '')
		time.sleep(FRESH_TIME)



def rl():
	q_table = build_q_table(N_STATES, ACTIONS)
	for episode in range(MAX_EPISODES):
		step_counter = 0
		S = 0
		is_terminated = False
		update_env(S, episode, step_counter)
		
		while not is_terminated:
			A = choose_action(S, q_table)
			S_, R = get_env_feedback(S, A)

			q_predict = q_table.ix[S, A]
			if S_ != 'terminal':
				q_target = R + GAMMA * q_table.iloc[S_, :].max()
			else:
				q_target = R
				is_terminated = True
			q_table.ix[S, A] += ALPHA * (q_target - q_predict)

			S = S_
			update_env(S, episode, step_counter + 1)
			step_counter += 1
	return q_table


if __name__ == "__main__":
	q_table = rl()
	print('\r\nQ-table:\n')
	print(q_table) 



