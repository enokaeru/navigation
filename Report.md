[//]: # (Image References)

[image1]: scores.png "Plot Scores"


# Report

## Learning  Algorithm
I adapt priorities replay buffer, dueling network, double dqn, Multi-step learning.
Hyper parameters are below, along to paper, git-hub repository.

| Hyper-parameter         | value    | reference                                                      |
|-------------------------|----------|----------------------------------------------------------------|
| Q network: hidden units | 256      | open ai gym baseline                                           |
| state_size              | 37       |                                                                |
| action_size             | 4        |                                                                |
| Discount factor         | 0.99     | Rainbow: Combining Improvements in Deep Reinforcement Learning |
| LR (optim Adam)         | 6.25E-05 | Rainbow: Combining Improvements in Deep Reinforcement Learning |
| epsilon (optim Adam)    | 1.25E-04 | Rainbow: Combining Improvements in Deep Reinforcement Learning |
| UPDATE_EVERY            | 4        | Rainbow: Combining Improvements in Deep Reinforcement Learning |
| BATCH_SIZE              | 64       | Rainbow: Combining Improvements in Deep Reinforcement Learning |
| prioritize alpha         | 0.6      | Rainbow: Combining Improvements in Deep Reinforcement Learning |
| prioritize beta          | 0.4→1    | Rainbow: Combining Improvements in Deep Reinforcement Learning |
| prioritize omega         | 0.5      | Rainbow: Combining Improvements in Deep Reinforcement Learning |
| priorities eps           | 1.00E-06 | udacity github                                                 |
| eps_start               | 1        | Rainbow: Combining Improvements in Deep Reinforcement Learning |
| eps_end                 | 0.02     | udacity github                                                 |
| multi-step              | 3.00     | Rainbow: Combining Improvements in Deep Reinforcement Learning |
| TAU                     | 1.00E-03 | CONTINUOUS  CONTROL  WITH  DEEP  REINFORCEMENT LEARNING         |


The model architectures are below along to dueling DQN.


|       Layer (type)            |   Output Shape      |    Param # |
|-------------------------------|---------------------| -----------|
|            Linear-1           |   [-1, 64, 64]        |   2,432   |
|            Linear-2           |    [-1, 64, 64]        |  4,160    |
|            Linear-3           |   [-1, 64, 256]        |  16,640   |
|            Linear-state_out   |   [-1, 64, 37]        |  9,509    |
|            Linear-action_out  |    [-1, 64, 4]        |   1,028   |
|            Linear-state_score |    [-1, 64, 1]        |      38   |

Total params: 33,807  <br>
Trainable params: 33,807 <br>
Non-trainable params: 0 <br>
<br>
Input size (MB): 0.01 <br>
Forward/backward pass size (MB): 0.21 <br>
Params size (MB): 0.13 <br>
Estimated Total Size (MB): 0.35 <br>
<br>
where Batch size is 64.

Output of the model are below.
```
output = state_score + action_out - action_out.mean()
```


## Plot of Rewords
821 episodes needed to solve the environment. 
I'm interested how much to improve the reward, so the training continued　even after solving.
The score decreased due to more learning.

![Plot Scores][image1]

## Ideas for Future Work
- I don't try to implement distribution DQN, Noise-Net from Rainbow.
I'd like to try to experiment those method, observe how effective they are.
- The agent should select action that quite depends on previous time. I think to be better that it adapt memory structure.
 I'd like to implement stacked input on DQN or LSTM on A2C.
