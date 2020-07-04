# tuteure

## prérequis :

- PySC2  https://github.com/deepmind/pysc2
- Starcraft2
- package python :
  - random
  - numpy
  - pandas
  - absl
  
## usage

- go to your PySC2 path then:

  `python -m pysc2.bin.agent --map Simple64 --agent pysc2.agents.QTable_Zerg.SmartAgent --agent2 pysc2.agents.QTable_Zerg.RandomAgent`

## sources :

https://medium.com/playing-sample-efficient-sc-ii/exploiting-intelligent-network-design-to-improve-sc2-agent-learning-efficiency-7c30dc049b8f

https://itnext.io/reinforcement-learning-with-raw-actions-and-observations-in-pysc2-af0b6fd8391f

https://github.com/deepmind/pysc2/pull/266

https://deepmind.com/blog/article/alphastar-mastering-real-time-strategy-game-starcraft-ii

https://github.com/deepmind/pysc2/commit/9dfbf1b050dcf0c71ef01fbce35efa6d79a3a21c

https://itnext.io/the-evolution-of-alphastar-cefff389b9d5

http://chris-chris.ai/2019/02/07/pysc2-tutorial3/

https://towardsdatascience.com/implementing-a-deepmind-baseline-starcraft-reinforcement-learning-agent-7b1ef6f41b52

http://bennycheung.github.io/adventures-in-deep-reinforcement-learning
