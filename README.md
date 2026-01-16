# MMRS Simulator

MMRS Simulator is an application developed for my masters thesis at Wroclaw University of Science and Technology. The program is designed to simulate a multiple mobile robot system (MMRS) environemt where robots execute task of achieveing specific locations. The app was developed in the idea of testing differnt ways and algorithms of avoiding deadlocks to optimize time needed to execute by all robots as well as entire system. By defining marked locations which are positions each robot must achieve the algorithm creates the path between the points using quadratic Bezier curves. The user is able to load a multiple number of robots via YAML file. The program was written in PyQt5.

Future development includes implementing full control abstraction that includes Supervisor as Discrete Event System (DES) which will communicate with multiple Robot controllsers (one for each robot) and the controllers will communicate directly to nobile robots which will work in Continuous Time System. Additionally to properly implement the deadlock avoidance the movemnt of the robots will be represented in Petri nets or Automata mathematical model. One more step is to develop an algorithm that will create new paths in real time to optimize time for robots waiting e.g. on crossroads.

## Key Features

- **Input:** Loading the number robots. - `agvs.yaml`.
- **Path creation:** Usage of Bezier curves to create robot paths.

## Quick Start

This section guides you through building PN-Car-Body-App from source code.

**Clone &rarr; Build &rarr; Run:**


```bash
git clone https://github.com/Mastej-Git/mmrs-sim.git
cd mmrs-sim
make setup-env
make run
```

## Tools

- **[Poetry](https://python-poetry.org/):** Python packaging and dependency management.
- **[Ruff](https://docs.astral.sh/ruff/):** Python linter and code formatter.

## Masters Thesis

Masters thesis location: `It does not exists yet`
