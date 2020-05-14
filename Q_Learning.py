#Code for Q-Learning for MountainCar-v0 using discrete timesteps



import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('MountainCar-v0')


Leaning_rate= 0.4
Discount= 0.95
games= 50000

SHOW= 5000

epsilon= 0.6
start_ep= 1
end_ep= games//2

decay_ep= epsilon /(end_ep- start_ep)

ep_reward =[]
avg_ep= {'ep': [], 'avg': [], 'min': [], 'max': []}

pos_space= np.linspace(-1.2, 0.6, 30)
vel_space= np.linspace(-0.07, 0.07, 30)

qtable= np.random.uniform(low=-2, high= 0, size= [30,30,3])



def discrete(observation):   
	pos, vel= observation
	pos_b= np.digitize(pos, pos_space)
	vel_b= np.digitize(vel,vel_space)

	return (pos_b, vel_b)




for i in range(games):

	this_ep= 0

	if i % SHOW== 0:
		print(i)
		renderr=True
	else:
		renderr= False



	done= False 
	obs= env.reset()
	the_discrete_state= discrete(env.reset())
	
	while not done:
		if np.random.random() > epsilon:
			action= np.argmax(qtable[the_discrete_state])
		else:
			action= np.random.randint(0, env.action_space.n)
		new_obs, reward, done, info= env.step(action)
		new_state= discrete(new_obs)
		this_ep += reward

		if renderr: 
			env.render()

		if not done:
			max_fut_q= np.max(qtable[new_state])
			current_q= qtable[the_discrete_state+ (action, )]
			new_q= (1- Leaning_rate)*current_q+ Leaning_rate*(reward+ Discount*max_fut_q)
			qtable[the_discrete_state+ (action, )]= new_q
		elif new_obs[0] >= env.goal_position:
			qtable[the_discrete_state+ (action, )]= 0 
			print("we made it on", i)

		the_discrete_state=new_state
	if end_ep >= i >= start_ep:
		epsilon -= decay_ep

	ep_reward.append(this_ep)

	if renderr:
		average_reward = sum(ep_reward[-SHOW:])/ len(ep_reward[-SHOW:])
		avg_ep['ep'].append(i)
		avg_ep['avg'].append(average_reward)
		avg_ep['min'].append(min(ep_reward[-SHOW:]))
		avg_ep['max'].append(max(ep_reward[-SHOW:]))

		print("episode: ", i)
		print("avg: ", average_reward)
		print("min_reward: ", min(ep_reward[-SHOW:]))
		print("max_reward: ", max(ep_reward[-SHOW:]))	



plt.plot(avg_ep['ep'], avg_ep['avg'], label= "avg")
plt.plot(avg_ep['ep'], avg_ep['min'], label= "min")
plt.plot(avg_ep['ep'], avg_ep['max'], label= "max")

plt.legend(loc=4)
plt.show()















