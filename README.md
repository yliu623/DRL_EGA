# DRL_EGA
## About
This is the companion code for the paper - An Empirically Grounding Analytics (EGA) Approach to Hog Farm Finishing Stage Management: Deep Reinforcement Learning as Decision Support and Managerial Learning Tool.
The paper is available at SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4617964

## Usage
Run myddpg_per.py for training and testing. To set a simulation environment for training and testing, modify the parameter in the code accordingly (TradingEnv(...)).
For hyperparameter tuning, run run.py.

## Credits
- The code structure is adapted from the code for the paper "Deep Hedging of Derivatives Using Reinforcement Learning" by Jay Cao, Jacky Chen, John Hull, and Zissis Poulos. (https://github.com/tdmdal/rl-hedge-2019)
- The implementation of prioritized experience replay buffer is taken from OpenAI Baselines.
