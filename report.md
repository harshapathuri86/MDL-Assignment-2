# Value Iteration

Number of iterations to converge: 115

If the enemy is in D, Indiana is at an advantage, so it tries to attack if possible.
If the enemy is in R, Indiana moves to a safer place, W, N, S, and performs possible actions at that state.

### State N
- If the enemy is in a Dormant state, Indiana tries to go down to attack it even if he has no material or arrows.
- If the enemy is in a Ready state, Indiana tries to stay away from the enemy and craft an arrow if he has some material. 

### State E
- As the chances to hit and shoot are high at this position, he tries to attack the enemy whenever possible.
- As the damage to the hit action is high, and the probability is highest in position E, he tries to hit the enemy with a blade when the health is full. 
- Indiana tries to shoot the enemy if he reached the arrow limit to make craft new arrows.

### State S
- If the enemy is in a Dormant state, Indiana tries to move up to attack it.
- Indiana tries to gather material if he has no arrows. 
- Stays at S, if the enemy is in R state to avoid the attack.

### State W
- If the enemy is in a Dormant state, Indiana tries to move right to attack it, as the probability of hitting/shooting is high in C, E compared to W.
- If the enemy is in Ready state, Indiana stays at W to avoid the attack and tries to shoot with the arrow.

### State C
- If Indiana has no material and arrows, he tries to gather material.
- Indiana tries to move right and attack the enemy if he has some arrows.
- If the enemy is in Ready state and Indiana has some material, he stays away from the attack and tries to craft arrows.
- If arrows are full and the enemy is in a ready state, moves left to avoid losing all arrows and tries to shoot.


## Simulation (W, 0, 0, D, 100)
```
(W,0,0,D,100):RIGHT
(C,0,0,D,100):RIGHT
(E,0,0,R,100):HIT
(E,0,0,R,100):HIT
(E,0,0,D,100):HIT
(E,0,0,D,50):HIT
(E,0,0,D,50):HIT
(E,0,0,R,50):HIT
(E,0,0,R,0):NONE
```

Indiana moves right to attack as the enemy is in a Dormant state.
At position E, as Indiana has 0 material and arrows, his best option is to hit with the blade.
He continues to hit with the blade till the enemy dies.


## Simulation (C, 2, 0, R, 100)
```
(C,0,2,R,100):UP
(N,0,2,R,100):STAY
(N,0,2,D,100):DOWN
(C,0,2,D,100):RIGHT
(E,0,2,D,100):HIT
(E,0,2,D,100):HIT
(E,0,2,D,100):HIT
(E,0,2,D,50):SHOOT
(E,0,1,R,25):SHOOT
(E,0,0,R,0):NONE
```
Indiana moves up to be safe as the enemy is in a Ready state.
Next, he moves down to attack as the enemy is in a Dormant state.
Next, moves right to attack as the chance of hitting with blade increases.
Fails to hit the enemy and hits again.
As enemy health decreased, Indiana shoots twice without a miss and kills it.

### CASE 1
> Indiana, now on the LEFT action at East Square, will go to West Square.
- Number of iterations to converge: 129
- Initially, when Indiana is at E, he tried to attack the enemy whenever possible, even if it is an R state. Because the enemy attacks Indiana even if he is in the C state, it is better to stay at E and attack.
- As we change the left action at state E when Indiana is at risk, he moves left to W and avoids the attack for sure.

### CASE 2
> The step cost of the STAY action is now zero.
- Number of iterations to converge: 104
- Indiana can stay in a position to avoid the attack with 0 costs. This makes Indiana more risk-averse as he can take more time to kill the enemy.
- Indiana avoids crafting and gathering if it is not essential and can just stay at S, N for protection.
- Previously, Indiana used to attack at position E even when the enemy is in a Ready state. Now, he can move left till W to stay there to avoid attacks and hit the enemy with the blade as the cost to stays is zero.

### CASE 3
> Change the value of gamma to 0.25
- Number of iterations to converge: 8
- Due to the small discount factor, the iterations converge fast.
- The policy is risk-averse compared to the original policy in some situations and risk-seeking in some other situations.
- This policy is not accurate as of the original policy. It lacks the long-term effects of the actions.