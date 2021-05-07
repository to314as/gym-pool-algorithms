# gym-pool-algorithms
A first selection of algorithms to tackle our new gym-pool envirnemnt! (You will need to install it: pip install pool-gym)
Can be seen as benchmark methods if you want to compete in the RL pool challange [which we hopefully get running ;)].

To replicate our experiments one may follow the general procedure of running:
python -m pool_alogrithms.models.train --algo ** --balls ** (--visualize)

To run the dqn algorithm run:
python -m pool_alogrithms.models.train --algo "dqn"
or run it directly within the folder (careful with the interdependiecies of the folder).

to run the REINFORCE algorithm:
python -m pool_alogrithms.models.train --algo "REINFORCE"
or run it directly within the folder (careful with the interdependiecies of the folder).

to run the q table algorithm:
python -m pool_alogrithms.models.train --algo q table
or run it directly within the folder (careful with the interdependiecies of the folder).

to run the brute force algorithm:
python -m pool_alogrithms.models.train --algo "brute force"
or run it directly within the folder (careful with the interdependiecies of the folder).

visualization.py is a utility file taht was used to generate some of the plots and make them more fun to look at. 
Some general plotting functionalities are also included in most training methods.

!Important note: using the --visualize is very nice to get an overview of what is happening but because of the way 
we were forced to design it (basically forcing python to sleep for 1/3 second before each shot) so that is humanly
possible to follow the actions of the agent drastically slows down training. A fraction of a second each turn might 
not sound too bad at first but it really does sum up. 1000000 times almost nothing is still something!

To understand the game and rules better looking into the gym-pool repo might be helpful.
Cheers and enjoy.