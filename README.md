<b>About the environment.</b>

The environment is available at:
https://gymnasium.farama.org/environments/toy_text/taxi/

<b>About the problem.</b>

An agent is trained in the Taxi environment with 500 states and 6 possible actions. The Taxi Problem involves navigating to passengers in a grid world, picking them up and dropping them off at one of four locations.

The states are defined by:
* the car's position (25 options),
* the passenger's position (4 options + option of being located in the car),
* the home's position (4 options).

The possible actions are:
* move down/up/right/left,
* attempt to pickup passenger,
* attempt to drop off passenger.

The rewards are: -
* -1 for a legal action,
* -10 for an illegal attempt to pickup or drop the passenger,
* +20 (and success) for dropping the passenger at the destination. 

<b>Selected algorithms and file structure.</b>

The agents are trained using Value Iteration with 10 steps and Q-Learning with 10000 episodes. The algorithm Value Iteration is implemented in the module  value_iteartion.py and called in the script value_iteartion.ipynb. Q-Learning is implemented in the module Q_learning.py and called in the script Q_learning.ipynb. The functions in the module play_episode.py are used for checking the policies by playing episodes (optionally, an episode is drawn). They are called in both scripts. The functions in the module decode_taxi.py decode the integer number of a state. They are used for printing an episode.


<b>Results.</b>

According to the results, all 10000 randomly generated episodes are won by both agents. The actions in the first 10 states show that the Q-learning agent shows variable behavior and chooses random actions in illegal states, while Value Iteration also allowed leaning actions in these states.

Visualizing one episode shows the car takes the shortest path.

Value Iteration runs faster, which shows that this algorithm is better adapted to this small environment.

<b>Possible improvements.</b>

As the environment is small and well-sructures, it is possible to implement the optimal policy by hand and use it to check in which measure the trained agents are sub-optimal. It is also possible to retrain several Q-learning agents and compare their mean rewards across a selection of episodes. 


<b>Feedback and additional questions.</b>

All questions about the source code should be adressed to its author Alexandre Aksenov:
* GitHub: Alexandre-aksenov
* Email: alexander1aksenov@gmail.com
