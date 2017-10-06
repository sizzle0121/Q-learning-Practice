import numpy as np
import pandas as pd
import time
import os

np.random.seed(2)

N_STATES = 16
ACTIONS = ['up', 'right', 'down', 'left']
#EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 20
FRESH_TIME = 0.2

def build_q_table(n_states, action):
	table = pd.DataFrame(
		np.zeros((n_states, len(action))),
		columns = action,
		)
	return table

def policy(state, q_table):
	state_actions = q_table.iloc[state, :]
	if state_actions.all() == 0:
		action_name = np.random.choice(ACTIONS)
	else:
		action_name = state_actions.argmax()
	return action_name

def get_env_feedback(S, A):
	if A == 'up':
		R = 0
		if S - 4 >= 0:
			S_ = S - 4
		else:
			S_ = S
	elif A == 'right':
		if S + 1 == 15:
			S_ = 'terminal'
			R = 1
		elif ((S + 1)%4 == 0):
			S_ = S
			R = 0
		else:
			S_ = S + 1
			R = 0
	elif A == 'down':
		if S + 4 == 15:
			S_ = 'terminal'
			R = 1
		elif S + 4 > 15:
			S_ = S
			R = 0
		else:
			S_ = S + 4
			R = 0
	elif A == 'left':
		R = 0
		if S%4 == 0:
			S_ = S
		else:
			S_ = S - 1
	return S_, R

def update_env(S, episode, step_counter):
	env = [0,0,0,0,
		   0,0,0,0,
		   0,0,0,0,
		   0,0,0,'#']
	if S == 'terminal':
		env[15] = '@'
	else:
		env[S] = '@'

	os.system('cls')
	for i in range(4):
		for j in range(4):
			print( env[j + 4*i] , end = ' ')
		print()
	if S == 'terminal':
		reaction = 'Episode %s: total steps = %s' % (episode+1, step_counter)
		print('\r{}'.format(reaction), end = '')
		time.sleep(2)
		print('\r                                ',end = '')
	else:
		time.sleep(FRESH_TIME)

def rl():
	q_table = build_q_table(N_STATES, ACTIONS)
	for episode in range(MAX_EPISODES):
		step_counter = 0
		S = 0
		is_terminated = False
		update_env(S, episode, step_counter)

		while not is_terminated:
			A = policy(S, q_table)
			S_, R = get_env_feedback(S, A)
			q_predict = q_table.ix[S, A]
			if S_ != 'terminal':
				q_target = R + GAMMA * q_table.iloc[S_,:].max()
			else:
				q_target = R
				is_terminated = True
			q_table.ix[S, A] += ALPHA * (q_target - q_predict)
			S = S_
			step_counter += 1
			update_env(S, episode, step_counter)

	return q_table

if __name__ == "__main__":
	q_table = rl()
	print('\r\nQ-table:\n')
	print(q_table)



