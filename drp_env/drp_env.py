import gym
import numpy as np
import sys
import copy
import os


from drp_env.state_repre import REGISTRY
from drp_env.EE_map import MapMake
from drp_env.gui_task import GUI_tasklist

sys.path.append(os.path.join(os.path.dirname(__file__), ''))

class DrpEnv(gym.Env):
	def __init__(self,
			agent_num,
			speed,
			start_ori_array,
			goal_array,
			visu_delay,
			state_repre_flag,
			time_limit,
			collision,
			map_name="map_3x3",
			reward_list={"goal": 100, "collision": -10, "wait": -10, "move": -1},
			task_flag=False,
			task_density=1.0
		  ):
		self.agent_num = agent_num
		self.n_agents = agent_num # for epymarl
		self.state_repre_flag = state_repre_flag
		self.map_name = map_name
		self.speed = speed
		self.visu_delay = visu_delay
		self.start_ori_array = start_ori_array
		self.goal_array = goal_array

		# reward
		self.r_goal = reward_list["goal"]
		self.r_coll = reward_list["collision"]
		self.r_wait = reward_list["wait"]
		self.r_move = reward_list["move"]

		# collision machnism
		self.collision = collision

		self.time_limit = time_limit
		self.task_density = task_density

		self.colli_distan_value = 0.1
		self.r_flag = 0
		self.flag_indicate = 0
		self.episode_account = 0

		# for tasklist
		self.task_completion = 0

		self.distance_from_start = np.zeros(self.agent_num)

		# create ee_env and pass self.variable
		self.ee_env = MapMake(self.agent_num, self.start_ori_array, self.goal_array, self.map_name)
		self.pos = self.ee_env.pos
		self.start_ori_array = self.ee_env.start_ori_array
		self.goal_array = self.ee_env.goal_array
		self.G = self.ee_env.G
		self.edge_labels = self.ee_env.edge_labels # unused

		self.current_goal  = [ None for i in range(self.agent_num)]

		self.obs_manager = REGISTRY[self.state_repre_flag](self)

		# create gym-like mdp elements
		self.n_nodes = len(self.G.nodes)
		self.n_actions = self.n_nodes
		self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.n_nodes)] * self.agent_num))
		
		obs_box = self.obs_manager.get_obs_box()
		self.observation_space = gym.spaces.Tuple(tuple([obs_box] * self.agent_num))

		self.log = {}

		# flag for tasklist
		self.is_tasklist = task_flag
		self.current_tasklist=[]
		self.assigned_tasks=[]#エージェントが割り当てられたタスク(未ピックを含む)
		self.assigned_list=[]#未実行のタスクとエージェントの割り当て表
		self.task_num = self.agent_num*2 # for tasklist, each agent can have 2 tasks at most
		#for rendering
		if self.is_tasklist:
			self.taskgui=GUI_tasklist()

	def get_obs(self):
		return self.obs

	def get_state(self): # unused
		return self.s

	def _get_avail_agent_actions(self, agent_id, n_actions):
		avail_actions = self.ee_env.get_avail_action_fun(
			self.obs[agent_id],
			self.current_start[agent_id],
			self.current_goal[agent_id],
			self.goal_array[agent_id],
		)

		# Allow idle agents (no remaining task to serve) to keep moving even if they reached their idle goal.
		if self.is_tasklist and agent_id < len(self.assigned_tasks):
			if len(self.assigned_tasks[agent_id]) == 0:
				at_node = (
					self.current_goal[agent_id] is None
					or self.current_goal[agent_id] == self.current_start[agent_id]
				)
				if at_node:
					current_node = self.current_start[agent_id]
					if current_node is None and len(self.obs) > agent_id:
						current_node = int(self.obs[agent_id][2])
					if current_node is not None:
						neighbors = list(self.G.neighbors(int(current_node)))
						# include wait action at current node
						neighbors.append(int(current_node))
						for nxt in neighbors:
							if nxt not in avail_actions:
								avail_actions.append(int(nxt))

		avail_actions = [int(a) for a in avail_actions if a is not None]
		if not avail_actions:
			fallback = self.current_start[agent_id]
			if fallback is None:
				fallback = 0
			avail_actions.append(int(fallback))
		# remove duplicates while preserving order
		seen = set()
		avail_actions = [a for a in avail_actions if not (a in seen or seen.add(a))]
		avail_actions_one_hot = np.zeros(n_actions)
		avail_actions_one_hot[avail_actions] = 1
		return avail_actions_one_hot, avail_actions
	
	def get_avail_agent_actions(self, agent_id, n_actions):
		return self._get_avail_agent_actions(agent_id, n_actions)

	def reset(self):
		# if goal and start are not assigned, randomly generate every episode    
		self.start_ori_array = copy.deepcopy(self.ee_env.input_start_ori_array)
		self.goal_array = copy.deepcopy(self.ee_env.input_goal_array)
		#print("self.start_ori_array", self.start_ori_array)
		if self.start_ori_array == []:
			self.ee_env.random_start()
			self.start_ori_array = self.ee_env.start_ori_array
		if self.goal_array == []:
			self.ee_env.random_goal()
			self.goal_array = self.ee_env.goal_array
		#print("self.start_ori_array after", self.start_ori_array)

		#initialize task list
		if self.is_tasklist:
			self.goal_array = copy.deepcopy(self.start_ori_array)
			self.current_tasklist=[]
			#self.assigned_tasks[i] is a task assigned to agent i
			self.assigned_tasks=[[] for _ in range(self.agent_num)] 
			self.assigned_list=[]
			self.alltasks = self.ee_env.create_tasklist(self.time_limit, self.agent_num, self.task_density)
			self.alltasks_flat = [task for step_tasks in self.alltasks for task in step_tasks]
			self.next_task_idx = 0

		#initialize obs
		self.obs = tuple(np.array([self.pos[self.start_ori_array[i]][0], self.pos[self.start_ori_array[i]][1], self.start_ori_array[i], self.goal_array[i]]) for i in range(self.agent_num))
		self.obs_current_chache = copy.deepcopy(self.obs)# used for calculating reward
		#initialize obs_one-hot
		self.obs_onehot = np.zeros((self.agent_num, self.n_nodes*2))
		for i in range(self.agent_num):
			self.obs_onehot[i][int(self.start_ori_array[i])] = 1 #current position
			self.obs_onehot[i][int(self.goal_array[i])+self.n_nodes] = 1 #current goal


		self.current_start = self.start_ori_array # [0,1]
		self.current_goal  = [None for _ in range(self.agent_num)]
		self.terminated    = [False for _ in range(self.agent_num)]

		self.distance_from_start = np.zeros(self.agent_num) # info
		self.wait_count = np.zeros(self.agent_num) # info

		self.reach_account = 0
		self.step_account = 0
		self.episode_account += 1

		# for tasklist
		self.task_completion = 0
		#print('Environment reset obs: \n', self.obs)
		self.last_info = None
		self.last_episode_reward = None
		self.last_termination_reason = None
		self.last_episode_steps = None

		obs = self.obs_manager.calc_obs()

		return obs
		

	def step(self, joint_action):

		if isinstance(joint_action, dict):
			task_assign = joint_action.get("task", None)
			joint_action = joint_action.get("pass", joint_action)

			filtered = [x for x in task_assign if x != -1]
			if len(filtered) != len(set(filtered)):
				raise ValueError("Error: Multiple agents assigned to the same task.")

		#transite env based on joint_action
		self.step_account += 1
		self.obs_current_chache = copy.deepcopy(self.obs)

		self.obs_prepare = []
		self.obs_onehot_prepare = copy.deepcopy(self.obs_onehot)
		self.current_start_prepare = copy.deepcopy(self.current_start)
		self.current_goal_prepare = copy.deepcopy(self.current_goal)
		# 1) first judge action_i whether available, to output !!!obs_prepare & obs_onehot_prepare!!!
		for i in range(self.agent_num):
			action_i = joint_action[i]  
			# 1) first judge action_i whether available, to output obs_prepare: 
			# if unavailable ⇢ obs_prepare.append( self.obs_old[i])
			#print("Avaible actions",self.get_avail_agent_actions(i, self.n_actions)[1])
			if action_i not in self._get_avail_agent_actions(i, self.n_actions)[1]:
				#print("This is not Avaible",i,action_i,self.get_avail_agent_actions(i, self.n_actions)[1])
				self.obs_prepare.append(self.obs_current_chache[i])
				#self.obs_onehot_prepare[i]= self.obs_onehot[i]

				self.wait_count[i] += 1

			# if action_i is current start node -> stop
			elif self.pos[int(action_i)][0]==self.obs[i][0] and self.pos[int(action_i)][1]==self.obs[i][1]:
				self.obs_prepare.append(self.obs_current_chache[i])
				self.wait_count[i] += 1
				#pbsのため，その場待機でもcurrent_goalをNoneのままでないように変更
				self.current_goal_prepare[i] = action_i
			# if available ⇢ obs_prepare update by obs_i_
			else:
				#self.joint_action_old[i] = joint_action[i]
				self.current_goal_prepare[i] = joint_action[i] #update 行き先ノード when avable action is taken
				obs_i = self.obs[i]
		
				#calculate current distance
				current_goal = list(self.pos[int(action_i)])
				current_x1,current_y1 = obs_i[0], obs_i[1]
				x = current_goal[0] - current_x1
				y = current_goal[1] - current_y1
				dist_to_cgoal = np.sqrt(np.square(x) + np.square(y))# the distance to current goal

				if dist_to_cgoal>self.speed:# move on edge
					current_x1 = round(current_x1+(self.speed*x/dist_to_cgoal), 2)
					current_y1 = round(current_y1+(self.speed*y/dist_to_cgoal), 2)
					obs_i_ = [round(current_x1,2), round(current_y1,2), obs_i[2], obs_i[3]]
					
					# for one-hot state
					x = list(self.pos[self.current_start[i]])[0] - current_x1
					y = list(self.pos[self.current_start[i]])[1] - current_y1
					dist_to_cstart = np.sqrt(np.square(x) + np.square(y))# the distance to current goal
					dist_to_cstart_rate = round(dist_to_cstart/(dist_to_cstart+dist_to_cgoal), 2)
					
					#print("self.obs_onehot_prepare before",self.obs_onehot_prepare )
					self.obs_onehot_prepare[i] = np.zeros((1, len(list(self.G.nodes()))*2))
					self.obs_onehot_prepare[i][int(action_i)] = dist_to_cstart_rate
					self.obs_onehot_prepare[i][int(self.current_start[i])] = 1-dist_to_cstart_rate
					self.obs_onehot_prepare[i][int(self.goal_array[i])+len(list(self.G.nodes()))] = 1 #current goal
					#print("self.obs_onehot_prepare after",self.obs_onehot_prepare )
					self.distance_from_start[i] += self.speed
				# arrive at node
				else:
					obs_i_ = [round(self.pos[int(action_i)][0],2), round(self.pos[int(action_i)][1],2), obs_i[2], obs_i[3]]
					
					# for one-hot state
					self.obs_onehot_prepare[i] = np.zeros((1, len(list(self.G.nodes()))*2))
					self.obs_onehot_prepare[i][int(action_i)] = 1
					self.obs_onehot_prepare[i][int(self.goal_array[i])+len(list(self.G.nodes()))] = 1 #current goal
					
					# update current_start only when arrive at node
					self.current_start_prepare[i] = int(action_i) #update 出発ノード when　行き先ノード　has been arrived
					self.current_goal_prepare[i] = None #update 行き先ノード when it has been arrived

					self.distance_from_start[i] += dist_to_cgoal

				self.obs_prepare.append(obs_i_)
		
		# 2) !!!obs_prepare & obs_onehot_prepare!!! を持って、
		# second judge whether to !!! obs & obs_onehot !!! according to collision happen
		collision_flag = self.ee_env.collision_detect(self.obs_prepare)
		info = {
			"goal": False,
			"collision": False,
			"timeup": False, # for epymarl
			"distance_from_start": None,
			"step": self.step_account,
			"wait": self.wait_count,
			"goal_account": self.reach_account,
			"1agent_goal_account": self.reach_account/self.agent_num,
			"task_completion": self.task_completion,
		}
		# happen
		if collision_flag==1:#collision
			#collision_reward=-1
			collision_reward = self.r_coll*self.speed
			if self.collision == "bounceback":
				self.terminated = [False for _ in range(self.agent_num)]
			else: # default -> self.collision == "terminated"
				self.terminated = [True for _ in range(self.agent_num)]
			info["collision"] = True
			#obs = self.obs_manager.calc_obs()
			ri_array = [collision_reward for _ in range(self.agent_num)]

			# Dynamically increase edge costs around collision to avoid reusing them
			try:
				pairs = getattr(self.ee_env, "last_collisions", [])
				edges_to_penalize = []
				for (ai, aj) in pairs:
					for k in (ai, aj):
						start = self.current_start[k] if k < len(self.current_start) else None
						next_goal = self.current_goal_prepare[k] if k < len(self.current_goal_prepare) else None
						if start is not None and next_goal is not None:
							# penalize the attempted traversal edge
							edges_to_penalize.append((int(start), int(next_goal)))
						elif start is not None:
							# fallback: penalize all incident edges to the collision node
							for nb in self.G.neighbors(int(start)):
								edges_to_penalize.append((int(start), int(nb)))
				if edges_to_penalize:
					self.add_edge_penalty(edges_to_penalize, penalty=5.0)
			except Exception:
				# do not break episode due to penalty bookkeeping issues
				pass
			
			# return obs, [collision_reward for _ in range(self.agent_num)], self.terminated, info 
			
		# not happen
		else: #non collision
			self.obs = tuple([np.array(i) for i in self.obs_prepare])
			self.obs_onehot = copy.deepcopy(self.obs_onehot_prepare)
			self.current_start = copy.deepcopy(self.current_start_prepare) 
			self.current_goal = copy.deepcopy(self.current_goal_prepare)

			team_reward = 0
			ri_array = []
			for i in range(self.agent_num):
				ri = self.reward(i)
				team_reward += ri
				ri_array.append(ri)
			
			if self.terminated == [True for _ in range(self.agent_num)]: # all reach goal
				#print("!!!all reach goal!!!")
				# info
				info["goal"] = True
			
			else:
				pass

			#obs = self.obs_manager.calc_obs()

		if self.is_tasklist:
			while len(self.current_tasklist) < self.task_num and self.next_task_idx < len(self.alltasks_flat):
				self.current_tasklist.append(self.alltasks_flat[self.next_task_idx])
				self.assigned_list.append(-1) # -1 means unassigned
				self.next_task_idx += 1

			# remove the task from the list if it has been completed
			for i in range(self.agent_num):
				pos_agenti = [self.obs[i][0],self.obs[i][1]]
				if len(self.assigned_tasks[i])>0:
					if str(pos_agenti)==str(self.pos[self.goal_array[i]]):
						if self.goal_array[i] == self.assigned_tasks[i][1]:
							self.assigned_tasks[i] = [] # remove the task from assigned_tasks
							self.task_completion += 1
			# assign tasks to agents
			for i in range(self.agent_num):
				if (self.assigned_tasks[i] == [] or i in self.assigned_list) and task_assign[i] != -1:
					self.assigned_tasks[i] = self.current_tasklist[task_assign[i]]
					self.goal_array[i] = self.assigned_tasks[i][0] # update goal to pick node
					if self.assigned_list[task_assign[i]] != -1:
						agj = self.assigned_list[task_assign[i]]
						
					self.assigned_list[task_assign[i]] = i # update assigned_list

			# update agent's start and goal
			for i in range(self.agent_num):
				pos_agenti = [self.obs[i][0],self.obs[i][1]]
				if len(self.assigned_tasks[i])>0:
					if str(pos_agenti)==str(self.pos[self.goal_array[i]]):
						#when agent i reach the pick node
						if self.goal_array[i]==self.assigned_tasks[i][0]:
							self.start_ori_array[i] = self.goal_array[i]
							self.goal_array[i] = self.assigned_tasks[i][1]
							try:
								idx = self.assigned_list.index(i)
								self.current_tasklist.pop(idx)
								self.assigned_list.pop(idx)
							except ValueError:
								print("ValueError: agent ", i, " 's assigned task is not in the current_tasklist")
						#when agent i reach the drop node
						elif self.goal_array[i]==self.assigned_tasks[i][1]:
							self.start_ori_array[i] = self.goal_array[i]
							#self.goal_array[i] = self.assigned_tasks[i][0]
						else:
							# print(self.goal_array[i], self.assigned_tasks[i])
							raise ValueError("Error in task execution")
				
				self.obs_prepare[i] = [self.obs[i][0], self.obs[i][1], self.start_ori_array[i], self.goal_array[i]]
				self.obs_onehot[i] = np.zeros((1, len(list(self.G.nodes()))*2))
				self.obs_onehot[i][int(self.start_ori_array[i])] = 1
				self.obs_onehot[i][int(self.goal_array[i])+len(list(self.G.nodes()))] = 1

			self.obs = tuple([np.array(i) for i in self.obs_prepare])
			while len(self.current_tasklist) < self.task_num and self.next_task_idx < len(self.alltasks_flat):
				self.current_tasklist.append(self.alltasks_flat[self.next_task_idx])
				self.assigned_list.append(-1) # -1 means unassigned
				self.next_task_idx += 1

		obs = self.obs_manager.calc_obs()
		# print(self.goal_array, self.obs[0][3],self.obs[1][3])

		# Check whether time is over
		if self.step_account >= self.time_limit:
			#print("!!!time up!!!")
			info["timeup"]= True
			self.terminated = [True for _ in range(self.agent_num)]

		info["distance_from_start"] = self.distance_from_start

		if all(self.terminated) is True:
			self.update_log(info)

		return obs, ri_array, self.terminated, info
	
	def update_log(self, info):
		log_episode = {}

		if info["goal"] is True:
			log_episode["result"] = "goal"
		elif info["collision"] is True:
			log_episode["result"] = "collision"
		elif info["timeup"] is True:
			log_episode["result"] = "timeup"
		else:
			log_episode["result"] = "exception"

		log_episode["termination_time"] = info["step"]
		log_episode["distance_from_start"] = info["distance_from_start"]

		self.log[self.episode_account] = log_episode

	def get_log(self, epi):
		return self.log[epi]

	def reward(self, i):
		pre_pos_agenti = [self.obs_current_chache[i][0],self.obs_current_chache[i][1]]
		pos_agenti = [self.obs[i][0],self.obs[i][1]]

		if self.is_tasklist: #ここから
			if self.start_ori_array[i] == self.goal_array[i]:
				r_i = 0
			else:
				if str(pos_agenti)==str(self.pos[self.goal_array[i]]): # at goal				
					if len(self.assigned_tasks[i])>0 : #first time to reach goal 
						r_i = self.r_goal
						self.reach_account += 1
					else: # stop at goal
						r_i = 0
						# self.distance_from_start[i] -= self.speed
			
				else: #at a general node 
					if pre_pos_agenti==pos_agenti: # stop at a general node 
						r_i = self.r_wait*self.speed
					else: # just move 
						r_i = self.r_move*self.speed

		else:
			if str(pos_agenti)==str(self.pos[self.goal_array[i]]): # at goal				
				if pre_pos_agenti!=pos_agenti : #first time to reach goal 
					r_i = self.r_goal
					self.reach_account += 1
					self.terminated[i] = True
				else: # stop at goal
					r_i = 0   
					# self.distance_from_start[i] -= self.speed
			
			else: #at a general node 
				if pre_pos_agenti==pos_agenti: # stop at a general node 
					r_i = self.r_wait*self.speed
				else: # just move 
					r_i = self.r_move*self.speed
		return r_i

	def render(self, mode='human'):
		self.ee_env.enable_rendering()
		self.ee_env.plot_map_dynamic(
			self.visu_delay,self.obs_current_chache,
			self.obs,self.goal_array,
			self.agent_num,
			self.current_goal,
			self.reach_account,
			self.step_account,
			self.episode_account,
			self.current_tasklist,
			self.assigned_tasks,
		) # a must be a angle !!!list!!!

		if self.is_tasklist:
			self.taskgui.show_tasklist(
				self.agent_num, 
				self.assigned_tasks, 
				self.current_tasklist,
				self.assigned_list
				)
		

	def close(self):
		reason = None
		if isinstance(self.last_termination_reason, str):
			reason = self.last_termination_reason
		if reason is None and isinstance(getattr(self, "last_info", None), dict):
			info = self.last_info
			if bool(info.get("collision")):
				reason = "collision"
			elif bool(info.get("goal")):
				reason = "task_completed"
			elif bool(info.get("timeup")):
				reason = "time_up"
		if reason is None:
			if isinstance(getattr(self, "terminated", None), list) and all(getattr(self, "terminated", []) or []):
				reason = "task_completed"
			elif getattr(self, "time_limit", None) is not None and getattr(self, "step_account", 0) >= int(self.time_limit):
				reason = "time_up"
		if reason is None and bool(getattr(self, "log", {}).get("collision", False)):
			reason = "collision"
		if reason is None:
			reason = "unknown"
		reward_val = getattr(self, "last_episode_reward", None)
		steps_val = getattr(self, "last_episode_steps", None)
		if reward_val is None:
			reward_str = "unknown"
		else:
			try:
				reward_str = f"{float(reward_val):.2f}"
			except Exception:
				reward_str = "unknown"
		if steps_val is None:
			steps_str = "unknown"
		else:
			try:
				steps_str = str(int(steps_val))
			except Exception:
				steps_str = "unknown"
		print(f"Environment CLOSE (reason={reason}, reward={reward_str}, steps={steps_str})")
		return None
    
	def get_pos_list(self):
		pos_list = []
		all_onehot_obs = np.array(self.obs_onehot)
		onehot_obs = all_onehot_obs[:, :self.n_nodes]

		# get all agent state and position
		for i, obs_i in enumerate(onehot_obs):
			edge_or_node = tuple([i for i, o in enumerate(obs_i) if o!=0])
			if len(edge_or_node)==1:
				node = edge_or_node[0]
				pos = {"type": "n", "pos": node}
				obs_i = np.array(obs_i)*self.agent_num
			else:
				edge = edge_or_node
				pos = {"type": "e", "pos": edge, "current_goal": self.current_goal[i], "current_start": self.current_start[i], "obs": obs_i}
			pos_list.append(pos)

		return pos_list

	def get_node_info(self):
		"""List nodes with positions."""
		return [{"id": int(n), "pos": list(self.pos[int(n)])} for n in self.G.nodes()]

	def get_edge_info(self):
		"""List edges with current weight, base weight, and accumulated penalty."""
		edges = []
		base = getattr(self.ee_env, "base_edge_weight", {})
		penalty = getattr(self.ee_env, "edge_penalty", {})
		for u, v, data in self.G.edges(data=True):
			key = (min(int(u), int(v)), max(int(u), int(v)))
			bw = float(base.get(key, float(data.get("weight", 0.0))))
			pen = float(penalty.get(key, 0.0))
			edges.append(
				{
					"u": int(u),
					"v": int(v),
					"weight": float(data.get("weight", 0.0)),
					"base_weight": bw,
					"penalty": pen,
				}
			)
		return edges

	def get_adjacency_list(self):
		"""Adjacency list (neighbors for each node)."""
		return {int(n): [int(nb) for nb in self.G.neighbors(n)] for n in self.G.nodes()}

	def add_edge_penalty(self, edges, penalty=5.0):
		"""Increase traversal cost for specified undirected edges [(u,v), ...]."""
		if hasattr(self.ee_env, "add_collision_penalty"):
			self.ee_env.add_collision_penalty(edges, penalty)
		return

	def get_path_length(self, start, goal):
		if start == goal:
			return 0
		else:
			return self.ee_env.get_path_length(start, goal)

	def set_1agent_info(self, pos, current_start, current_goal, goal_array):
		self.obs = tuple(np.array([pos[0], pos[1], self.obs[0][2], self.obs[0][3]]) for _ in range(1))
		self.current_start[0] = current_start
		self.current_goal[0] = current_goal
		self.goal_array[0] = goal_array
		self.step_account = 0

		return
