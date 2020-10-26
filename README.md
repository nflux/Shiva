# Shiva

Shiva is built to be simulation-engine agnostic, its framework is abstracted to support various types of observation and 
actions spaces with different environment settings and number of agents. Additionally, Shiva is designed to support 
distributed processing across a large number of servers to support learning in complex environments with large 
observation or action spaces where multiple agents need to converge to a team policy. At the moment, Shiva supports 
popular reinforcement and imitation learning algorithms such as Deep Q-Network (DQN), Deep Deterministic Policy Gradient 
(DDPG), Proximal Policy Optimizations (PPO), Multi Agent Deep Deterministic Policy Gradient (MADDPG), Dataset 
Aggregation (DAGGER) method in addition to a few customized and hybrid model-based algorithms that leverage the dynamics 
of the environment to converge to at a faster rate. The framework is built to enable researchers to design and 
experiment with new algorithms and be able to test them at scale in different environments and scenarios with minimum 
setup on the infrastructure.

Get started with the Installation and then thru the Quickstart to see how to run a session. The Tutorial section goes in 
more details about Shiva to familiarize with it's components and then be able to extend new algorithms.

## Table of Content

1. [Requirements and Installation](./shiva/docs/Getting-Started.md)
2. [Quickstart](./shiva/docs/Quick-Start.md)
3. Tutorial
    * Project Layout
    * Classes
        * [Algorithm](./shiva/docs/Algorithms.md)
        * [Agent](./shiva/docs/Agents.md)
        * [MetaLearner](./shiva/docs/Metalearners.md)
        * [Learner](./shiva/docs/Learners.md)
        * [Environment](./shiva/docs/Environments.md)
        * [Network](./shiva/docs/Networks.md)
        * [Buffer](./shiva/docs/Buffers.md)
        * [Admin](./shiva/docs/Admin.md)
    * [Configuration files](./shiva/docs/Config-Files.md)
    * Example Environments
4. How to extend Shiva
    * [UnitTests](./shiva/docs/UnitTests.md)
    * [Creating a new algorithm](./shiva/docs/Extending-Algorithm.md)
    * Creating a new environment wrapper

## Benchmarks

You can use these benchmarks to test if changes made to Shiva were improvements.

## Restrictions

If you would like to contribute to Shiva, we would like you to do so by providing your own implementations of the abstract modules to maintain stability. If you have difficulties with any of the existing modules please raise an issue on the repository.

## Credits

* nFlux
   * Seyyed Sajjadi
   * Andrew Miller
   * Ezequiel Donovan
   * Jorge Martinez
   * Travis Hamm
   * Daniel Tellier
   * Joshua Kristanto
* ICT / DoD
* CSUN NSF Grant # 1842386

## License

[Apache License 2.0](LICENSE)
