# Linear Programming

### Matrix A

[comment]: <> (Explain procedure of making A matrix)

- Matrix A is of shape `600*1936`
- Every row represents a valid state in the MDP
- Every column represents a valid action at a particular state
- For every transition action `A` at state `S1`, we calculated the probability `p` of reaching for each possible next state using the `action(action_type, state)` function. It returns the expected cost and a list of the next states with their probability
- As it is the inflow for state `S2`, we subtracted it for `S2`. As it is the outflow for state `S1`, we added it. If `S1` and `S2` are the same, we ignored it as it causes a self-loop
  > `A[S2A][S2]-=p` `A[S1A][S1]+=p`

### Policy 

- 

[comment]: <> ( Explain procedure of finding the policy and analyze the results. )


### Multiple Policies
[comment]: <> (Can there be multiple policies? Why? What changes can you make in your code to generate another policy? &#40; Do not paste code snippets, explain the changes in terms of how they will affect the A matrix, R vector, alpha vector etc.&#41; )

- Suppose we change our r vector (reward) by changing the step cost or adding the final reward or changing the stay action cost, or changing the cost of any particular action, there is a chance that we will get a different policy.
- Suppose we change our A matrix by changing the flow, like by changing the probabilities of actions leading to different states, there is a chance that we will get a different policy.
- Suppose we change our alpha vector by changing probabilities of other states also being a start state. In that case, there is a chance that we will get a different policy because randomness is introduced to the model. Whenever we run the algorithm, we have some probability of any of the possible start states can be our initial state.