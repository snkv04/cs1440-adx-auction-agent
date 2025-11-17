from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator, CONFIG
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment
from typing import Set, Dict

USER_SEGMENT_PMF = CONFIG['user_segment_pmf']


class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        super().__init__()
        self.name = 'Agent 67'
        self.on_new_game()
        self.min_profit_margin = 0.10
        self.target_effective_reach = 1.05
        self.a = 4.08577
        self.b = 3.08577
        self.delta = 0.5
        self.old_demand = {seg: 0.0 for seg in USER_SEGMENT_PMF.keys()}
        self.new_demand = {seg: 0.0 for seg in USER_SEGMENT_PMF.keys()}

    def on_new_game(self) -> None:
        # Reset demand tracking for new game
        self.old_demand = {seg: 0.0 for seg in USER_SEGMENT_PMF.keys()}
        self.new_demand = {seg: 0.0 for seg in USER_SEGMENT_PMF.keys()}

    def increment_demand_values(self, campaign: Campaign) -> None:
        target_segment = campaign.target_segment
        reach = campaign.reach

        # Find all atomic segments that contain the target segment
        matching_atomic_segments = []
        for atomic_seg in USER_SEGMENT_PMF.keys():
            if target_segment.issubset(atomic_seg):
                matching_atomic_segments.append(atomic_seg)

        if not matching_atomic_segments:
            return

        # Calculate total PMF for matching segments
        total_pmf = sum(USER_SEGMENT_PMF[seg] for seg in matching_atomic_segments)
        if total_pmf == 0:
            portion_per_segment = reach / len(matching_atomic_segments)
            for seg in matching_atomic_segments:
                self.new_demand[seg] += portion_per_segment
        else:
            for seg in matching_atomic_segments:
                proportion = USER_SEGMENT_PMF[seg] / total_pmf
                self.new_demand[seg] += reach * proportion

    def get_atomic_segment_cpc_estimate(self, target_segment: MarketSegment) -> float:
        demand = self.old_demand[target_segment]
        population = CONFIG['market_segment_pop'][target_segment]
        current_day = self.get_current_day()

        if population * current_day == 0:
            return self.delta

        ratio = demand / (population * current_day) + self.delta
        return ratio

    def get_segment_cpc_estimate(self, target_segment: MarketSegment) -> float:
        # Find all atomic segments that contain the target segment
        matching_atomic_segments = []
        for atomic_seg in USER_SEGMENT_PMF.keys():
            if target_segment.issubset(atomic_seg):
                matching_atomic_segments.append(atomic_seg)

        if not matching_atomic_segments:
            return self.delta

        # Collect CPC estimates and populations for matching atomic segments
        cpc_estimates = []
        populations = []
        total_population = 0
        for atomic_seg in matching_atomic_segments:
            cpc = self.get_atomic_segment_cpc_estimate(atomic_seg)
            population = CONFIG['market_segment_pop'][atomic_seg]

            cpc_estimates.append(cpc)
            populations.append(population)
            total_population += population

        # Compute weighted average based on populations (which includes delta)
        weighted_sum = sum(cpc * pop for cpc, pop in zip(cpc_estimates, populations))
        weighted_avg = weighted_sum / total_population

        return weighted_avg

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        bids = {}
        Q_A = self.get_quality_score()

        for campaign in campaigns_for_auction:
            # Increment demand values for this campaign (for the next day)
            self.increment_demand_values(campaign)

            R = campaign.reach
            target_segment = campaign.target_segment

            # Find maximum possible cost
            estimated_cpc = self.get_segment_cpc_estimate(target_segment)
            K_max = R * estimated_cpc

            # Get target effective bid
            rho_factor = self.target_effective_reach - self.min_profit_margin
            if rho_factor <= 0:
                continue
            B_target = K_max / rho_factor

            # Get actual bid from target effective bid
            B_bid = B_target * Q_A
            final_bid = self.clip_campaign_bid(campaign, B_bid)

            bids[campaign] = final_bid

        return bids

    def derivative_effective_reach(self, x: int, R: int) -> float:
        if R == 0:
            return 0.0

        u = self.a * float(x) / R - self.b
        return 2 / (R * (1 + u ** 2))

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()

        active_campaigns = self.get_active_campaigns()
        current_day = self.get_current_day()

        for campaign in active_campaigns:
            R = campaign.reach
            B = campaign.budget
            K_c = self.get_cumulative_cost(campaign)
            x_c = self.get_cumulative_reach(campaign)

            # Spread remaining budget over remaining days
            remaining_budget = B - K_c
            days_left = campaign.end_day - current_day + 1
            L_C = min(remaining_budget, remaining_budget / days_left)
            L_C = max(0.01, L_C)

            # Set up bid entries
            bid_entries = set()
            d_rho_dx = self.derivative_effective_reach(x_c, R)
            marginal_revenue = B * d_rho_dx
            bid_per_item = min(10.0, max(0.01, marginal_revenue))
            target_segment = campaign.target_segment
            bid_entry = Bid(
                bidder=self,
                auction_item=target_segment,
                bid_per_item=bid_per_item,
                bid_limit=L_C
            )
            bid_entries.add(bid_entry)

            # Create the BidBundle for the campaign
            bundle = BidBundle(
                campaign_id=campaign.uid,
                limit=L_C,
                bid_entries=bid_entries
            )
            bundles.add(bundle)

        # Replace old_demand with new_demand for the next day, and rollover the remaining demand to the next day
        self.old_demand = self.new_demand.copy()
        self.new_demand = {seg: max(0, self.new_demand[seg] - CONFIG['market_segment_pop'][seg])
                           for seg in USER_SEGMENT_PMF.keys()}

        return bundles


if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=10)
