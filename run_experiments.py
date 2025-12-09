import json
import sys
import os
from contextlib import contextmanager
from my_agent import MyNDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier2.my_agent import Tier2NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator


@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Default values
DEFAULT_GAMMA = 1.05
DEFAULT_TAU = 0.9
DEFAULT_OMEGA = 0.1

# Values to test
GAMMA_VALUES = [1.0, 1.125, 1.25, 1.5, 2.0]
TAU_VALUES = [0.3, 0.5, 0.7, 0.9, 1.1]
OMEGA_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0]

NUM_SIMULATIONS = 100


def run_experiment(opponent_class, opponent_name: str,
                   gamma: float = DEFAULT_GAMMA,
                   tau: float = DEFAULT_TAU,
                   omega: float = DEFAULT_OMEGA) -> float:
    agent = MyNDaysNCampaignsAgent(gamma=gamma, tau=tau, omega=omega)
    opponents = [opponent_class(name=f"{opponent_name} {i + 1}") for i in range(9)]
    test_agents = [agent] + opponents

    simulator = AdXGameSimulator()
    with suppress_stdout():
        profits = simulator.run_simulation(agents=test_agents, num_simulations=NUM_SIMULATIONS)

    return profits[agent.name] / NUM_SIMULATIONS


def run_gamma_experiments():
    results = {}

    for gamma in GAMMA_VALUES:
        key = str(gamma)
        results[key] = {}

        print(f"\n{'='*60}")
        print(f"Testing gamma = {gamma}")
        print('='*60)

        print(f"\nRunning against Tier1 agents...")
        results[key]["tier1"] = run_experiment(Tier1NDaysNCampaignsAgent, "Tier1", gamma=gamma)
        print(f"Average profit vs Tier1: {results[key]['tier1']:.2f}")

        print(f"\nRunning against Tier2 agents...")
        results[key]["tier2"] = run_experiment(Tier2NDaysNCampaignsAgent, "Tier2", gamma=gamma)
        print(f"Average profit vs Tier2: {results[key]['tier2']:.2f}")

    return results


def run_tau_experiments():
    results = {}

    for tau in TAU_VALUES:
        key = str(tau)
        results[key] = {}

        print(f"\n{'='*60}")
        print(f"Testing tau = {tau}")
        print('='*60)

        print(f"\nRunning against Tier1 agents...")
        results[key]["tier1"] = run_experiment(Tier1NDaysNCampaignsAgent, "Tier1", tau=tau)
        print(f"Average profit vs Tier1: {results[key]['tier1']:.2f}")

        print(f"\nRunning against Tier2 agents...")
        results[key]["tier2"] = run_experiment(Tier2NDaysNCampaignsAgent, "Tier2", tau=tau)
        print(f"Average profit vs Tier2: {results[key]['tier2']:.2f}")

    return results


def run_omega_experiments():
    results = {}

    for omega in OMEGA_VALUES:
        key = str(omega)
        results[key] = {}

        print(f"\n{'='*60}")
        print(f"Testing omega = {omega}")
        print('='*60)

        print(f"\nRunning against Tier1 agents...")
        results[key]["tier1"] = run_experiment(Tier1NDaysNCampaignsAgent, "Tier1", omega=omega)
        print(f"Average profit vs Tier1: {results[key]['tier1']:.2f}")

        print(f"\nRunning against Tier2 agents...")
        results[key]["tier2"] = run_experiment(Tier2NDaysNCampaignsAgent, "Tier2", omega=omega)
        print(f"Average profit vs Tier2: {results[key]['tier2']:.2f}")

    return results


def print_summary(name: str, values: list, results: dict):
    print(f"\n{name} Experiments Summary:")
    print(f"{name:<10} {'vs Tier1':<15} {'vs Tier2':<15}")
    print("-" * 40)
    for val in values:
        key = str(val)
        print(f"{val:<10} {results[key]['tier1']:<15.2f} {results[key]['tier2']:<15.2f}")


def main():
    all_results = {}

    # Run gamma experiments
    print("\n" + "=" * 60)
    print("GAMMA EXPERIMENTS (tau={}, omega={})".format(DEFAULT_TAU, DEFAULT_OMEGA))
    print("=" * 60)
    all_results["gamma"] = run_gamma_experiments()

    # Run tau experiments
    print("\n" + "=" * 60)
    print("TAU EXPERIMENTS (gamma={}, omega={})".format(DEFAULT_GAMMA, DEFAULT_OMEGA))
    print("=" * 60)
    all_results["tau"] = run_tau_experiments()

    # Run omega experiments
    print("\n" + "=" * 60)
    print("OMEGA EXPERIMENTS (gamma={}, tau={})".format(DEFAULT_GAMMA, DEFAULT_TAU))
    print("=" * 60)
    all_results["omega"] = run_omega_experiments()

    # Save all results to JSON
    output_file = "experiment_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summaries
    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print('='*60)

    print_summary("Gamma", GAMMA_VALUES, all_results["gamma"])
    print_summary("Tau", TAU_VALUES, all_results["tau"])
    print_summary("Omega", OMEGA_VALUES, all_results["omega"])


if __name__ == "__main__":
    main()
