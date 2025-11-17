from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment 
from typing import Set, Dict

class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        # TODO: fill this in (if necessary)
        super().__init__()
        self.name = "Agent 6 7"
        self.on_new_game()
        self.counts = {
            "start_before": 0,
            "end_before": 0,
            "start_after": 0,
            "end_after": 0,
            "start_at": 0,
            "end_at": 0,
            "total": 0
        }

    def on_new_game(self) -> None:
        # self.current_day = 1 # they keep track of it for us
        pass

    def get_ad_bids(self) -> Set[BidBundle]:
        # TODO: fill this in
        bundles = set()

        return bundles

    def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:
        for campaign in campaigns_for_auction:
            if campaign._start < self.current_day:
                self.counts["start_before"] += 1
            if campaign._end < self.current_day:
                self.counts["end_before"] += 1
            if campaign._start > self.current_day:
                self.counts["start_after"] += 1
            if campaign._end > self.current_day:
                self.counts["end_after"] += 1
            if campaign._start == self.current_day:
                self.counts["start_at"] += 1
            if campaign._end == self.current_day:
                self.counts["end_at"] += 1
            self.counts["total"] += 1

        # TODO: fill this in 
        bids = {}

        return bids

if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=10)

    print(f"counts = {test_agents[0].counts}")
