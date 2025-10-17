# Reinforcement Learning (RL)

Reinforcement Learning (RL) is a fascinating area of Machine Learning where an **agent learns to make decisions** by interacting with an **environment** to achieve a **goal**. Unlike supervised learning (where labels are given), RL is all about **trial and error**, learning from **feedback** in the form of rewards or penalties.


## Learning Objectives

By the end of this section, you will:
- Understand the **core concepts** behind Reinforcement Learning.
- Explore **Markov Decision Processes (MDPs)** and their role in RL.
- Implement **value-based** and **policy-based** learning algorithms.
- Train agents using **Q-Learning, SARSA, and DQN**.
- Apply **Actor-Critic and Policy Gradient** methods.
- Work with **OpenAI Gym** to simulate real-world RL environments.
- Experiment with **advanced RL algorithms** like **PPO, DDPG, and SAC**.


## 📘 Module Structure

| No. | Notebook | Description |
|-----|-----------|--------------|
| 00 | [Introduction_to_Reinforcement_Learning.ipynb](./00-Introduction_to_Reinforcement_Learning.ipynb) | Overview of RL concepts, agents, environments, and reward mechanisms. |
| 01 | [Markov_Decision_Processes.ipynb](./01-Markov_Decision_Processes.ipynb) | Understanding MDPs — the mathematical foundation of RL. |
| 02 | [Dynamic_Programming_Basics.ipynb](./02-Dynamic_Programming_Basics.ipynb) | Value Iteration and Policy Iteration explained and implemented. |
| 03 | [Monte_Carlo_Methods.ipynb](./03-Monte_Carlo_Methods.ipynb) | Using Monte Carlo simulations to learn value functions. |
| 04 | [Temporal_Difference_Learning.ipynb](./04-Temporal_Difference_Learning.ipynb) | Bridging Monte Carlo and Dynamic Programming with TD methods. |
| 05 | [Q_Learning.ipynb](./05-Q_Learning.ipynb) | Implementing the classic Q-learning algorithm. |
| 06 | [SARSA_Algorithm.ipynb](./06-SARSA_Algorithm.ipynb) | On-policy learning through SARSA. |
| 07 | [Deep_Q_Networks_DQN.ipynb](./07-Deep_Q_Networks_DQN.ipynb) | Applying Deep Neural Networks to approximate Q-values. |
| 08 | [DQN_Improvements_DDQN_and_Prioritized.ipynb](./08-DQN_Improvements_DDQN_and_Prioritized.ipynb) | Enhancing DQN with Double DQN and Prioritized Experience Replay. |
| 09 | [Policy_Gradients_Basics.ipynb](./09-Policy_Gradients_Basics.ipynb) | Introducing policy-based methods and stochastic policies. |
| 10 | [REINFORCE_Algorithm.ipynb](./10-REINFORCE_Algorithm.ipynb) | Implementing the fundamental policy gradient algorithm. |
| 11 | [Actor_Critic_Methods.ipynb](./11-Actor_Critic_Methods.ipynb) | Combining value-based and policy-based methods. |
| 12 | [Advanced_Actor_Critic_A3C_A2C.ipynb](./12-Advanced_Actor_Critic_A3C_A2C.ipynb) | Parallel actor-critic algorithms for faster convergence. |
| 13 | [Deep_Deterministic_Policy_Gradient_DDPG.ipynb](./13-Deep_Deterministic_Policy_Gradient_DDPG.ipynb) | Continuous action control using DDPG. |
| 14 | [Proximal_Policy_Optimization_PPO.ipynb](./14-Proximal_Policy_Optimization_PPO.ipynb) | Training robust agents with PPO. |
| 15 | [Soft_Actor_Critic_SAC.ipynb](./15-Soft_Actor_Critic_SAC.ipynb) | Entropy-regularized RL for exploration and stability. |
| 16 | [RL_in_OpenAI_Gym.ipynb](./16-RL_in_OpenAI_Gym.ipynb) | Setting up and exploring RL environments with Gym. |
| 17 | [Training_Agent_on_CartPole.ipynb](./17-Training_Agent_on_CartPole.ipynb) | Training a DQN agent to balance the CartPole. |
| 18 | [Playing_Atari_with_DQN.ipynb](./18-Playing_Atari_with_DQN.ipynb) | Building a DQN to play Atari games. |
| 19 | [RL_for_Continuous_Control.ipynb](./19-RL_for_Continuous_Control.ipynb) | RL techniques for continuous control tasks. |
| 20 | [RL_in_Robotics_Simulation.ipynb](./20-RL_in_Robotics_Simulation.ipynb) | Applying RL in robotic control and physics simulations. |

## 🧩 Key Concepts Covered

- Agent-Environment Interaction
- Rewards, Policies, and Value Functions  
- Bellman Equations and Optimality  
- Exploration vs Exploitation  
- On-Policy vs Off-Policy Learning  
- Policy Gradient and Actor-Critic Methods  
- Deep RL Architectures (DQN, PPO, SAC, etc.)  
- Applications in Games, Robotics, and Control Systems  


## ⚙️ Prerequisites

Before diving into RL:
- Solid understanding of **Python and NumPy**.
- Basic knowledge of **Neural Networks (Keras/PyTorch)**.
- Familiarity with **Supervised Learning** concepts.


## 🚀 Practical Outcomes

After completing this module, you’ll be able to:
- Design and train your own **RL agents**.
- Implement **Q-learning, DQN, PPO, and A3C** from scratch.
- Evaluate and fine-tune agent performance.
- Apply RL to **real-world control problems and simulations**.

## 🧭 Next Steps

Continue to the next section : **[09-MLOps_and_Deployment](../09-MLOps_and_Deployment/README.md)** where you’ll learn how to **deploy**, **monitor**, and **scale ML and RL models** in production environments.

**Author:** *Deep Learning Curriculum — Practical ML Series*  
**Maintained by:** *Sibasish Padhihari*
