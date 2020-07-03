# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 20:28:12 2020

@author: gglaive
"""

import random
import numpy as np
import pandas as pd
import os
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop

class QLearningTable:
  def __init__(self, actions, learning_rate=0.1, reward_decay=0.9):
    self.actions = actions
    self.learning_rate = learning_rate
    self.reward_decay = reward_decay
    self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
    

  def choose_action(self, observation, e_greedy=0.9):
    self.check_state_exist(observation)
    if np.random.uniform() < e_greedy:
      state_action = self.q_table.loc[observation, :]
      action = np.random.choice(
          state_action[state_action == np.max(state_action)].index)
    else:
      action = np.random.choice(self.actions)
    return action

  def learn(self, s, a, r, s_):
    self.check_state_exist(s_)
    q_predict = self.q_table.loc[s, a]
    if s_ != 'terminal':
      q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()
    else:
      q_target = r
    self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)

  def check_state_exist(self, state):
    if state not in self.q_table.index:
      self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), 
                                                   index=self.q_table.columns, 
                                                   name=state))
      self.q_table.to_csv("selfplay.csv")


class Agent(base_agent.BaseAgent):
	actions = ("do_nothing",
			"harvest_minerals",
			"build_spawning_pool",
			"train_zergling",
			"train_queen",
			"train_overlord",
			"do_injection",
			"attack")

	def get_my_units_by_type(self, obs, unit_type):
		return [unit for unit in obs.observation.raw_units
		  if unit.unit_type == unit_type 
		  and unit.alliance == features.PlayerRelative.SELF]

	def get_enemy_units_by_type(self, obs, unit_type):
		return [unit for unit in obs.observation.raw_units
		  if unit.unit_type == unit_type 
		  and unit.alliance == features.PlayerRelative.ENEMY]

	def get_my_completed_units_by_type(self, obs, unit_type):
		return [unit for unit in obs.observation.raw_units
		  if unit.unit_type == unit_type 
		  and unit.build_progress == 100
		  and unit.alliance == features.PlayerRelative.SELF]

	def get_enemy_completed_units_by_type(self, obs, unit_type):
		return [unit for unit in obs.observation.raw_units
		  if unit.unit_type == unit_type 
		  and unit.build_progress == 100
		  and unit.alliance == features.PlayerRelative.ENEMY]
		
	def unit_type_is_selected(self, obs, unit_type):
		if (len(obs.observation.single_select) > 0 and
			obs.observation.single_select[0].unit_type == unit_type):
			return True
		if (len(obs.observation.multi_select) > 0 and
			obs.observation.multi_select[0].unit_type == unit_type):
			return True
		return False

	def get_distances(self, obs, units, xy):
		units_xy = [(unit.x, unit.y) for unit in units]
		return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

	def step(self, obs):
		super(Agent, self).step(obs)
		if obs.first():
			hatchery = self.get_my_units_by_type(
					obs, units.Zerg.Hatchery)[0]
			self.base_top_left = (hatchery.x < 32)

	def do_nothing(self, obs):
		return actions.RAW_FUNCTIONS.no_op()

	def harvest_minerals(self, obs):
		drones = self.get_my_units_by_type(obs, units.Zerg.Drone)
		idle_drones = [drone for drone in drones if drone.order_length == 0]
		if len(idle_drones) > 0:
			mineral_patches = [unit for unit in obs.observation.raw_units
					  if unit.unit_type in [
							  units.Neutral.BattleStationMineralField,
							  units.Neutral.BattleStationMineralField750,
							  units.Neutral.LabMineralField,
							  units.Neutral.LabMineralField750,
							  units.Neutral.MineralField,
							  units.Neutral.MineralField750,
							  units.Neutral.PurifierMineralField,
							  units.Neutral.PurifierMineralField750,
							  units.Neutral.PurifierRichMineralField,
							  units.Neutral.PurifierRichMineralField750,
							  units.Neutral.RichMineralField,
							  units.Neutral.RichMineralField750]]
			drone = random.choice(idle_drones)
			distances = self.get_distances(obs, mineral_patches, (drone.x, drone.y))
			mineral_patch = mineral_patches[np.argmin(distances)] 
			return actions.RAW_FUNCTIONS.Harvest_Gather_unit(
					"now", drone.tag, mineral_patch.tag)
		return actions.RAW_FUNCTIONS.no_op()

	def build_spawning_pool(self, obs):
		spawning_pools = self.get_my_units_by_type(obs, units.Zerg.SpawningPool)
		drones = self.get_my_units_by_type(obs, units.Zerg.Drone)
		if (len(spawning_pools) == 0 and obs.observation.player.minerals >= 200 and len(drones) > 0):
			spawning_pool_xy = (22, 21) if self.base_top_left else (35, 45)
			distances = self.get_distances(obs, drones, spawning_pool_xy)
			drone = drones[np.argmin(distances)]
			return actions.RAW_FUNCTIONS.Build_SpawningPool_pt("now", drone.tag, spawning_pool_xy)
		return actions.RAW_FUNCTIONS.no_op()

	def train_overlord(self, obs):
		larvaes = self.get_my_units_by_type(obs, units.Zerg.Larva)
		if (obs.observation.player.minerals >= 100 and len(larvaes) > 0):
			return actions.RAW_FUNCTIONS.Train_Overlord_quick("now", random.choice(larvaes).tag)
		return actions.RAW_FUNCTIONS.no_op()

	def train_zergling(self, obs):
		completed_spawning_pools = self.get_my_completed_units_by_type(obs, units.Zerg.SpawningPool)
		larvaes = self.get_my_units_by_type(obs, units.Zerg.Larva)
		free_supply = (obs.observation.player.food_cap - obs.observation.player.food_used)
		if (len(completed_spawning_pools) > 0 and obs.observation.player.minerals >= 50
		  and free_supply > 0 and len(larvaes) > 0):
			return actions.RAW_FUNCTIONS.Train_Zergling_quick("now", random.choice(larvaes).tag)
		return actions.RAW_FUNCTIONS.no_op()

	def train_queen(self, obs):
		completed_hatcheries = self.get_my_completed_units_by_type(obs, units.Zerg.Hatchery)
		free_supply = (obs.observation.player.food_cap - 
				 obs.observation.player.food_used)
		if (len(completed_hatcheries) > 0 and obs.observation.player.minerals >= 150 and free_supply >= 2):
			return actions.RAW_FUNCTIONS.Train_Queen_quick("now", completed_hatcheries[0].tag)
		return actions.RAW_FUNCTIONS.no_op()

	def do_injection(self, obs):
		completed_hatcheries = self.get_my_completed_units_by_type(obs, units.Zerg.Hatchery)
		queens = self.get_my_units_by_type(obs, units.Zerg.Queen)
		if (len(completed_hatcheries) > 0 and len(queens) > 0): #random.choice(queens).energy >= 25
			return actions.RAW_FUNCTIONS.Effect_InjectLarva_unit("now", random.choice(queens).tag,
														random.choice(completed_hatcheries).tag)
		return actions.RAW_FUNCTIONS.no_op()

	def attack(self, obs):
#		zerglings = self.get_my_units_by_type(obs, units.Zerg.Zergling)
		zerglings = [unit.tag for unit in obs.observation.raw_units
				   if unit.unit_type == units.Zerg.Zergling
				   and unit.alliance == features.PlayerRelative.SELF]
		if len(zerglings) > 0:
			attack_xy = (38, 44) if self.base_top_left else (19, 23)
			x_offset = random.randint(-4, 4)
			y_offset = random.randint(-4, 4)
			return actions.RAW_FUNCTIONS.Attack_pt(
				"now", zerglings, (attack_xy[0] + x_offset, attack_xy[1] + y_offset))
		return actions.RAW_FUNCTIONS.no_op()


class RandomAgent(Agent):
  def step(self, obs):
    super(RandomAgent, self).step(obs)
    action = random.choice(self.actions)
    return getattr(self, action)(obs)


class SmartAgent(Agent):
	def __init__(self):
		super(SmartAgent, self).__init__()
		self.qtable = QLearningTable(self.actions)
		self.new_game()

	def reset(self):
		super(SmartAgent, self).reset()
		self.new_game()

	def new_game(self):
		self.base_top_left = None
		self.previous_state = None
		self.previous_action = None

	def get_state(self, obs):
		drones = self.get_my_units_by_type(obs, units.Zerg.Drone)
		idle_drones = [drone for drone in drones if drone.order_length == 0]
		hatcheries = self.get_my_units_by_type(obs, units.Zerg.Hatchery)
		overlords = self.get_my_units_by_type(obs, units.Zerg.Overlord)
		spawning_pools = self.get_my_units_by_type(obs, units.Zerg.SpawningPool)
		completed_spawning_pools = self.get_my_completed_units_by_type(
				obs, units.Zerg.SpawningPool)
		zerglings = self.get_my_units_by_type(obs, units.Zerg.Zergling)
		queens = self.get_my_units_by_type(obs, units.Zerg.Queen)

		free_supply = (obs.observation.player.food_cap - 
				 obs.observation.player.food_used)
		can_afford_overlord = obs.observation.player.minerals >= 100
		can_afford_spawning_pool = obs.observation.player.minerals >= 200
		can_afford_zergling = obs.observation.player.minerals >= 50
		can_afford_queen = obs.observation.player.minerals >= 150

		enemy_drones = self.get_enemy_units_by_type(obs, units.Zerg.Drone)
		enemy_idle_drones = [drone for drone in enemy_drones if drone.order_length == 0]
		enemy_hatcheries = self.get_enemy_units_by_type(
				obs, units.Zerg.Hatchery)
		enemy_overlords = self.get_enemy_units_by_type(
				obs, units.Zerg.Overlord)
		enemy_spawning_pools = self.get_enemy_units_by_type(obs, units.Zerg.SpawningPool)
		enemy_completed_spawning_pools = self.get_enemy_completed_units_by_type(
				obs, units.Zerg.SpawningPool)
		enemy_zerglings = self.get_enemy_units_by_type(obs, units.Zerg.Zergling)
		enemy_queens = self.get_my_units_by_type(obs, units.Zerg.Queen)

		return (len(hatcheries),
		  len(drones),
		  len(idle_drones),
		  len(overlords),
		  len(spawning_pools),
		  len(completed_spawning_pools),
		  len(zerglings),
		  len(queens),
		  free_supply,
		  can_afford_overlord,
		  can_afford_spawning_pool,
		  can_afford_zergling,
		  can_afford_queen,
		  len(enemy_hatcheries),
		  len(enemy_drones),
		  len(enemy_idle_drones),
		  len(enemy_overlords),
		  len(enemy_spawning_pools),
		  len(enemy_completed_spawning_pools),
		  len(enemy_zerglings),
		  len(enemy_queens))

	def step(self, obs):
		super(SmartAgent, self).step(obs)
		state = str(self.get_state(obs))
		action = self.qtable.choose_action(state)
		if self.previous_action is not None:
			self.qtable.learn(self.previous_state,
					 self.previous_action,
					 obs.reward,
					 'terminal' if obs.last() else state)
		self.previous_state = state
		self.previous_action = action
		return getattr(self, action)(obs)


def main(unused_argv):
  agent1 = SmartAgent()
  agent2 = SmartAgent()
  try:
    with sc2_env.SC2Env(
        map_name="Simple64",
        players=[sc2_env.Agent(sc2_env.Race.zerg), 
                 sc2_env.Agent(sc2_env.Race.zerg)],
        agent_interface_format=features.AgentInterfaceFormat(
            action_space=actions.ActionSpace.RAW,
            use_raw_units=True,
            raw_resolution=64,
        ),
        step_mul=48,
        game_steps_per_episode=14000,
        disable_fog=True,
    ) as env:
      run_loop.run_loop([agent1, agent2], env, max_episodes=500)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  app.run(main)