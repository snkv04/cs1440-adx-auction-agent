import random
from my_agent import MyNDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator

PMPD_UPPER_BOUNDS = [0.5] * 10
PMPD_LOWER_BOUNDS = [-0.25] * 10

TERPD_UPPER_BOUNDS = [1.38] * 10
TERPD_LOWER_BOUNDS = [0.5] * 10

DELTA_LOWER_BOUND = 0.0
DELTA_UPPER_BOUND = 1.0

NUM_TRIALS = 200
NUM_SIMULATIONS = 100

OUTPUT_FILE = 'random-search.txt'


def sample_uniform(upper_bound, lower_bound):
    r = random.random()
    scaling_factor = upper_bound - lower_bound
    return r * scaling_factor + lower_bound


def random_search_all_params(num_trials=NUM_TRIALS, output_file=OUTPUT_FILE):
    profits_and_hyperparameters = []
    for i in range(num_trials):
        PMPD = []
        TERPD = []
        for j in range(10):
            PMPD.append(sample_uniform(PMPD_UPPER_BOUNDS[j], PMPD_LOWER_BOUNDS[j]))

        for j in range(10):
            TERPD.append(sample_uniform(TERPD_UPPER_BOUNDS[j], TERPD_LOWER_BOUNDS[j]))

        delta = sample_uniform(DELTA_UPPER_BOUND, DELTA_LOWER_BOUND)

        agent = MyNDaysNCampaignsAgent(delta, PMPD, TERPD)
        test_agents = [agent] + \
                      [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

        simulator = AdXGameSimulator()
        profits = simulator.run_simulation(agents=test_agents, num_simulations=NUM_SIMULATIONS)
        agent_profit = profits[agent.name]
        profits_and_hyperparameters.append((agent_profit, PMPD, TERPD, delta))
        print(f"Trial {i} completed successfully, average_profit: {agent_profit / NUM_SIMULATIONS}")

    profits_and_hyperparameters = sorted(profits_and_hyperparameters, key=lambda x: -x[0])
    with open(output_file, 'w') as file:
        for item in profits_and_hyperparameters:
            file.write(f"{item}\n")


def random_search_delta(num_trials=NUM_TRIALS, num_simulations=NUM_SIMULATIONS, output_file=OUTPUT_FILE):
    profits_and_hyperparameters = []
    for i in range(num_trials):
        delta = sample_uniform(DELTA_UPPER_BOUND, DELTA_LOWER_BOUND)
        agent = MyNDaysNCampaignsAgent(delta=delta)
        test_agents = [agent] + \
                      [Tier1NDaysNCampaignsAgent(name=f"Random Agent {i + 1}") for i in range(6)] + \
                      [Tier1NDaysNCampaignsAgent(name=f"Conservative Agent {i + 1}") for i in range(3)]

        simulator = AdXGameSimulator()
        profits = simulator.run_simulation(agents=test_agents, num_simulations=num_simulations)
        agent_profit = profits[agent.name]
        profits_and_hyperparameters.append((agent_profit, delta))
        print(f"Trial {i} completed successfully, average_profit: {agent_profit / NUM_SIMULATIONS}")
        profits_and_hyperparameters = sorted(profits_and_hyperparameters, key=lambda x: -x[0])
        with open(output_file, 'w') as file:
            for item in profits_and_hyperparameters:
                file.write(f"{item}\n")


if __name__ == "__main__":
    random_search_delta()
