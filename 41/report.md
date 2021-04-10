# Linear Programming

### Matrix A

- Matrix A is of shape `600*1936`
- Every row represents a valid state in the MDP
- Every column represents a valid action at a particular state
- For every transition action `A` at state `S1`, we calculated the probability `p` of reaching for each possible next state using the `action(action_type, state)` function. It returns the expected cost and a list of the next states with their probability
- As it is the inflow for state `S2`, we subtracted it for `S2`. As it is the outflow for state `S1`, we added it. If `S1` and `S2` are the same, we ignored it as it causes a self-loop
  > `A[S2A][S2]-=p` `A[S1A][S1]+=p`

### Policy 

- After solving the linear program, we get `X`, the optimal solution.
- We can get the policy by working in a reverse manner of creating `R`(reward) vector.
- We have to find the actions which correspond to the same state and the action which gives the highest value is the optimal action for that state.
- For example,
  > If we have `S1_Ac1,S1_Ac2,S1_Ac3,S2_Ac2,S2_Ac5 ...` in the `X`, we find the maximum value in `S1_Ac1,S1_Ac2,S1_Ac3`. If it is of `S1_Ac2`, then `Ac2` is the best action in the state `S1`. The array of all the best actions in the order of states gives us the policy.
- The agent only crafts arrows or gathers material when it is essential like if the enemy's health is high. He doesn't do it i the enemy's health is low as he can kill it with a throw of a blade.
- He tries to move towards enemy when it is dormant state so he can attack it with high precesion.
- If the enemy is in ready state, he moves away from enemy. If he can go to North or South, he chooses based on the available material and arrows.


### Multiple Policies

- Suppose we change our r vector (reward) by changing the step cost or adding the final reward or changing the stay action cost, or changing the cost of any particular action, there is a chance that we will get a different policy.
- Suppose we change our A matrix by changing the flow, like by changing the probabilities of actions leading to different states, there is a chance that we will get a different policy.
- Suppose we change our alpha vector by changing probabilities of other states also being a start state. In that case, there is a chance that we will get a different policy because randomness is introduced to the model. Whenever we run the algorithm, we have some probability of any of the possible start states can be our initial state.
