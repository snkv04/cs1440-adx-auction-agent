from typing import Set, Dict, Tuple
import math

from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier2.my_agent import Tier2NDaysNCampaignsAgent
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment
from agt_server.local_games.adx_arena import AdXGameSimulator, CONFIG

USER_SEGMENT_PMF = CONFIG['user_segment_pmf']
MARKET_SEGMENT_POP = CONFIG['market_segment_pop']

# Hyperparameters
DEFAULT_GAMMA = 1.05        # Scaling factor for anticipated demand increase (gamma >= 1)
DEFAULT_OMEGA = 0.1         # Baseline hidden demand factor per population (omega)
DEFAULT_TAU = 0.9           # Lower bound for target reach ratio (tau)
DEFAULT_EPSILON = 0.001     # Small amount to bid above the expected order statistic
DEFAULT_RANK_OFFSET = 1e6   # Offset added to computed rank (makes bids more conservative)

# Effective reach constants
DEFAULT_A = 4.08577
DEFAULT_B = 3.08577


class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self,
                 name='956f1f38-b6cb-43ea-9811-a2c0764a67d9',
                 gamma=DEFAULT_GAMMA,
                 omega=DEFAULT_OMEGA,
                 tau=DEFAULT_TAU,
                 epsilon=DEFAULT_EPSILON,
                 a=DEFAULT_A,
                 b=DEFAULT_B,
                 rank_offset=DEFAULT_RANK_OFFSET):
        super().__init__()
        self.name = name

        # Hyperparameters
        self.gamma = gamma
        self.omega = omega
        self.tau = tau
        self.epsilon = epsilon
        self.a = a
        self.b = b
        self.rank_offset = rank_offset

        # Target reaches for each campaign (computed during campaign bidding)
        self.campaign_target_reach: Dict[int, int] = {}

        self.on_new_game()

    def on_new_game(self) -> None:
        self.demand: Dict[Tuple[int, MarketSegment], float] = {}
        for day in range(1, 11):
            for atomic_seg in USER_SEGMENT_PMF.keys():
                pop = MARKET_SEGMENT_POP[atomic_seg]
                self.demand[(day, atomic_seg)] = self.omega * pop

        self.undecided_campaigns: Set[Campaign] = set()
        self.campaign_target_reach = {}

    def get_atomic_segments(self, segment: MarketSegment) -> list:
        matching = []
        for atomic_seg in USER_SEGMENT_PMF.keys():
            if segment.issubset(atomic_seg):
                matching.append(atomic_seg)
        return matching

    def increment_demand_values(self, campaign: Campaign) -> None:
        """Update demand values for a campaign we didn't win"""
        target_segment = campaign.target_segment
        reach = campaign.reach
        start_day = campaign.start_day
        end_day = campaign.end_day
        num_days = end_day - start_day + 1

        atomic_segments = self.get_atomic_segments(target_segment)
        if not atomic_segments:
            return

        segment_pop = MARKET_SEGMENT_POP[target_segment]
        if segment_pop == 0:
            return

        # Distribute reach across days and atomic segments proportionally
        reach_per_day = reach / num_days
        for day in range(start_day, end_day + 1):
            for atomic_seg in atomic_segments:
                atomic_pop = MARKET_SEGMENT_POP[atomic_seg]
                proportion = atomic_pop / segment_pop
                demand_increment = reach_per_day * proportion
                self.demand[(day, atomic_seg)] += demand_increment

    def clip_rank(self, rank: int) -> int:
        return max(1, min(10, rank))

    def get_rank_to_beat(self, day: int, atomic_segment: MarketSegment, target_reach: float) -> int:
        pop = MARKET_SEGMENT_POP[atomic_segment]
        demand = self.demand[(day, atomic_segment)]

        if demand <= 0:
            return 9

        demand_per_agent = demand / 9.0
        if demand_per_agent <= 0:
            return 9

        users_we_dont_need = pop - target_reach
        rank = int(users_we_dont_need / demand_per_agent) + 1
        return self.clip_rank(rank)

    def get_user_bid_from_rank(self, rank: int) -> float:
        """Assumes a uniform distribution of bids for each agent"""
        adjusted_rank = self.clip_rank(rank + self.rank_offset)
        return 1.0 - adjusted_rank / 10.0

    def get_expected_cost_for_users(self, num_users: float, bid_to_beat: float) -> float:
        return bid_to_beat * num_users

    def get_scaled_demand(self, day: int, atomic_segment: MarketSegment, campaign_start: int) -> float:
        """Gamma_{e, a} = gamma^{e - c_l} * D_{e, a}, where e is in [c_l, c_r]"""
        pop = MARKET_SEGMENT_POP[atomic_segment]
        base_demand = self.demand[(day, atomic_segment)]

        # Scale by gamma^(days since campaign start)
        days_since_start = day - campaign_start
        scaling = self.gamma ** days_since_start

        return scaling * base_demand

    def get_expected_cost_for_campaign(self, campaign: Campaign, target_reach: int) -> float:
        target_segment = campaign.target_segment
        start_day = campaign.start_day
        end_day = campaign.end_day
        num_days = end_day - start_day + 1

        atomic_segments = self.get_atomic_segments(target_segment)
        if not atomic_segments:
            return 0.0

        segment_pop = MARKET_SEGMENT_POP[target_segment]
        if segment_pop == 0:
            return 0.0

        target_per_day = target_reach / num_days
        total_cost = 0.0
        for day in range(start_day, end_day + 1):
            for atomic_seg in atomic_segments:
                # t(c, a) for this atomic segment = proportional target reach
                atomic_pop = MARKET_SEGMENT_POP[atomic_seg]
                proportion = atomic_pop / segment_pop
                atomic_target = target_per_day * proportion

                # Use scaled demand for future cost estimation
                scaled_demand = self.get_scaled_demand(day, atomic_seg, start_day)

                # Compute rank needed with scaled demand
                demand_per_agent = scaled_demand / 9.0 if scaled_demand > 0 else 0.001
                users_we_dont_need = atomic_pop - atomic_target
                rank = int(users_we_dont_need / demand_per_agent) + 1 if demand_per_agent > 0 else 9
                rank = self.clip_rank(rank)

                # Expected cost using scaled rank
                bid = self.get_user_bid_from_rank(rank)
                cost = self.get_expected_cost_for_users(atomic_target, bid)
                total_cost += cost

        return total_cost

    def effective_reach(self, impressions: int, reach_goal: int) -> float:
        if reach_goal == 0:
            return 0.0

        ratio = impressions / reach_goal
        return (2 / self.a) * (math.atan(self.a * ratio - self.b) - math.atan(-self.b))

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        bids = {}
        quality_score = self.get_quality_score()

        # Process undecided campaigns from previous day, by updating demand for campaigns we didn't win
        active_campaign_ids = {c.uid for c in self.get_active_campaigns()}
        for campaign in self.undecided_campaigns:
            if campaign.uid not in active_campaign_ids:
                # We didn't win this campaign, so update the corresponding demand values
                self.increment_demand_values(campaign)
        self.undecided_campaigns.clear()

        for campaign in campaigns_for_auction:
            reach_goal = campaign.reach

            # Step 1: Find target reach t(c)
            # Search in range [ceil(tau * R), ceil(1.38 * R)]
            min_target = math.ceil(self.tau * reach_goal)
            max_target = math.ceil(1.38 * reach_goal)

            best_target = min_target
            best_profit = float('-inf')

            for target in range(min_target, max_target + 1):
                rho = self.effective_reach(target, reach_goal)
                expected_cost = self.get_expected_cost_for_campaign(campaign, target)
                profit = rho * reach_goal - expected_cost

                if profit > best_profit:
                    best_profit = profit
                    best_target = target

            # Store target reach for use in user auction
            self.campaign_target_reach[campaign.uid] = best_target

            # Step 2: Find bid b(c) ensuring 10% profit margin
            # b(c) = k(c) / (0.9 * rho(t(c), R))
            rho = self.effective_reach(best_target, reach_goal)
            expected_cost = self.get_expected_cost_for_campaign(campaign, best_target)

            assert rho > 0, f"Effective reach is 0 for campaign {campaign.uid}"
            target_budget = expected_cost / (0.9 * rho)

            # Clip bid to valid range
            final_bid = self.clip_campaign_bid(campaign, target_budget)
            bids[campaign] = final_bid

        # Add these campaigns to undecided campaigns for next day's demand update
        self.undecided_campaigns = set(campaigns_for_auction)

        return bids

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        active_campaigns = self.get_active_campaigns()
        current_day = self.get_current_day()

        if not active_campaigns:
            return bundles

        # Step 1: Compute target reach t(a) for each atomic segment
        atomic_target_reach: Dict[MarketSegment, float] = {seg: 0.0 for seg in USER_SEGMENT_PMF.keys()}
        campaign_atomic_targets: Dict[int, Dict[MarketSegment, float]] = {}

        for campaign in active_campaigns:
            # Get target reach for this campaign (default to reach goal if not set)
            total_target = self.campaign_target_reach.get(campaign.uid, campaign.reach)

            # Account for already acquired impressions
            cumulative_reach = self.get_cumulative_reach(campaign)
            remaining_target = max(0, total_target - cumulative_reach)

            # Pace evenly over remaining days
            days_left = campaign.end_day - current_day + 1
            if days_left <= 0:
                continue
            target_today = remaining_target / days_left

            # Distribute across atomic segments proportionally
            target_segment = campaign.target_segment
            atomic_segments = self.get_atomic_segments(target_segment)
            segment_pop = MARKET_SEGMENT_POP[target_segment]

            campaign_atomic_targets[campaign.uid] = {}
            if segment_pop > 0 and atomic_segments:
                for atomic_seg in atomic_segments:
                    atomic_pop = MARKET_SEGMENT_POP[atomic_seg]
                    proportion = atomic_pop / segment_pop
                    atomic_target = target_today * proportion

                    atomic_target_reach[atomic_seg] += atomic_target
                    campaign_atomic_targets[campaign.uid][atomic_seg] = atomic_target

        # Step 2: Determine bid for each atomic segment
        # Bid(a) = E[B(a)_{[r]}] + epsilon where r = rank to beat
        atomic_bids: Dict[MarketSegment, float] = {}

        for atomic_seg in USER_SEGMENT_PMF.keys():
            target = atomic_target_reach[atomic_seg]
            if target <= 0:
                atomic_bids[atomic_seg] = 0.01  # Minimum bid
                continue

            rank = self.get_rank_to_beat(current_day, atomic_seg, target)
            expected_bid = self.get_user_bid_from_rank(rank)
            # Bid slightly above to win
            atomic_bids[atomic_seg] = min(1.0, expected_bid + self.epsilon)

        # Step 3: Create BidBundles for each campaign
        for campaign in active_campaigns:
            if campaign.uid not in campaign_atomic_targets:
                continue

            target_segment = campaign.target_segment
            atomic_segments = self.get_atomic_segments(target_segment)

            bid_entries = set()
            total_limit = 0.0

            for atomic_seg in atomic_segments:
                atomic_target = campaign_atomic_targets[campaign.uid].get(atomic_seg, 0)
                if atomic_target <= 0:
                    continue

                bid_per_item = atomic_bids[atomic_seg]
                bid_limit = bid_per_item * atomic_target

                bid_entry = Bid(
                    bidder=self,
                    auction_item=atomic_seg,
                    bid_per_item=max(0.01, min(1.0, bid_per_item)),
                    bid_limit=max(0.01, bid_limit)
                )
                bid_entries.add(bid_entry)
                total_limit += bid_limit

            if not bid_entries:
                continue

            bundle = BidBundle(
                campaign_id=campaign.uid,
                limit=total_limit,
                bid_entries=bid_entries
            )
            bundles.add(bundle)

        return bundles


if __name__ == "__main__":
    agent = MyNDaysNCampaignsAgent()
    # Test offline against TA agents
    test_agents = [agent] + [Tier2NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    simulator = AdXGameSimulator()
    profits = simulator.run_simulation(agents=test_agents, num_simulations=10)
    print(profits[agent.name])
