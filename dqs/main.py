## outline

# 1. generate training example
#  - pick a xi
#  - generate the data for that xi
#    - convert to equations somehow?
#  - return single axis of certain length, correct answer
#
# 2. DQN
#  - preprocess
#    - regenerate representation w/ henkel + svd
#    - get starting xi w/ least-squares regress.
#  - RL scaffold
#    - define state and action spaces
#    - keep track of state history for sampling?
#    - forward pass on Q(s,a,theta) function
#    - epsilon-greedy selection of next action
#    - take action, get next state
#    - create target function
#    - evaluate loss function? (or is this already done?)
#    - backpropogation on Q
#  - Q function
#    - what architecture?
#      - input: actions vector, xi, residuals
#      - output: value of actions

# 3. modern training improvements
#  - double Q network to separate estimation and selection?
#  - smarter sampling of history?
