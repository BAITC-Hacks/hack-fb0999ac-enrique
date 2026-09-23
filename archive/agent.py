"""Budget-aware tariff campaign agent.

Historical transitions are used only to rank pilot hypotheses. Final campaign
decisions use noisy pilot observations with an uncertainty penalty.
"""

from pathlib import Path

import numpy as np
import pandas as pd


_CHANNEL_MULTIPLIERS = {
    "push": 0.50,
    "sms": 0.65,
    "digital_ads": 0.85,
    "call": 1.20,
}
_PILOT_STD_PER_CUSTOMER = 0.804
_ONE_SIDED_Z = 1.28  # approximately a 90% one-sided confidence bound
_MAX_CAMPAIGNS = 10
_MAX_CAMPAIGN_SIZE = 5_000
_MAX_PILOTS = 20
_PILOT_SIZE = 150


class Agent:
    def _historical_priors(self, tariff_codes):
        """Return rough SMS-scale uplift priors from the provided history."""
        history_path = Path(__file__).resolve().parent / "data" / "change_tariff.csv"
        try:
            history = pd.read_csv(history_path)
            history["arpu_segment"] = pd.cut(
                history["AVG_ARPU_PREV_3M"],
                bins=[-np.inf, 1_000, 5_000, np.inf],
                labels=["LOW", "MID", "HIGH"],
            )
            history = history.loc[history["AVG_ARPU_PREV_3M"] >= 100].copy()
            history["relative_change"] = (
                (history["AVG_ARPU_NEXT_3M"] - history["AVG_ARPU_PREV_3M"])
                / history["AVG_ARPU_PREV_3M"]
            ).clip(-1, 3)
            history = history.dropna(subset=["arpu_segment"])
            history["target_change"] = np.where(
                history["tariff_plan_code_from"] != history["tariff_plan_code_to"],
                history["relative_change"],
                0.0,
            )

            grouped = history.groupby(
                ["tariff_plan_code_from", "arpu_segment", "tariff_plan_code_to"],
                observed=True,
            )["target_change"].agg(["mean", "std", "count"])
            source_counts = history.groupby(
                ["tariff_plan_code_from", "arpu_segment"], observed=True
            ).size()

            priors = {}
            for (source, segment, target), row in grouped.iterrows():
                if target not in tariff_codes or source not in tariff_codes:
                    continue
                denominator = int(source_counts.get((source, segment), 0))
                if denominator == 0:
                    continue

                # The observed history contains transitions, not campaign
                # assignments. Treat its destination share as a weak prior.
                mean = float(row["mean"])
                prior_mean = mean * float(row["count"]) / denominator
                within_sd = float(row["std"]) if pd.notna(row["std"]) else 0.0
                prior_sd = max(0.12, min(0.35, within_sd / np.sqrt(max(row["count"], 1))))
                priors[(source, str(segment), target)] = {
                    "mean": prior_mean * _CHANNEL_MULTIPLIERS["sms"],
                    "sd": prior_sd * _CHANNEL_MULTIPLIERS["sms"],
                    "history_n": int(row["count"]),
                }
            return priors
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            # The agent can still explore using pilots if the optional history
            # file is unavailable in a judging environment.
            return {}

    @staticmethod
    def _posterior(prior_mean, prior_sd, observations):
        prior_var = max(prior_sd**2, 1e-8)
        precision = 1.0 / prior_var
        weighted_sum = prior_mean * precision
        for observed_ratio, sample_size, channel in observations:
            multiplier = _CHANNEL_MULTIPLIERS[channel]
            # Normalize each pilot to the SMS scale so channel measurements can
            # be compared. The environment adds the same absolute noise to all.
            normalized = observed_ratio * _CHANNEL_MULTIPLIERS["sms"] / multiplier
            noise_sd = (
                _PILOT_STD_PER_CUSTOMER
                / np.sqrt(max(sample_size, 1))
                * _CHANNEL_MULTIPLIERS["sms"]
                / multiplier
            )
            obs_precision = 1.0 / max(noise_sd**2, 1e-8)
            precision += obs_precision
            weighted_sum += normalized * obs_precision
        return weighted_sum / precision, np.sqrt(1.0 / precision)

    @staticmethod
    def _audience_cells(profile):
        cells = (
            profile.groupby(["current_tariff", "arpu_segment"], observed=True)
            .agg(
                n=("ID_NUMBER", "size"),
                mean_arpu=("predicted_arpu", "mean"),
            )
            .reset_index()
        )
        return cells

    def act(self, env):
        profile = env.customer_profile
        tariff_codes = set(env.tariffs["tariff_plan_code"].astype(str))
        prices = env.tariffs.set_index("tariff_plan_code")["price_tariff"].to_dict()
        channels = env.channels
        priors = self._historical_priors(tariff_codes)
        cells = self._audience_cells(profile)

        # Use existing ARPU segments and current tariffs as simple, interpretable
        # strata. Very small cells cannot support a useful pilot or campaign.
        candidates = []
        for cell in cells.itertuples(index=False):
            source = str(cell.current_tariff)
            segment = str(cell.arpu_segment)
            audience_n = int(cell.n)
            if audience_n < 180 or source not in prices:
                continue

            choices = []
            for target in sorted(tariff_codes):
                if target == source or prices[target] < prices[source]:
                    continue
                prior = priors.get((source, segment, target))
                if not prior or prior["mean"] <= 0:
                    continue
                # Estimate the value left after a 150-person SMS pilot. The
                # historical estimate ranks hypotheses only; pilots decide.
                pilot_n = min(_PILOT_SIZE, audience_n)
                scalable_n = min(audience_n, _MAX_CAMPAIGN_SIZE) - pilot_n
                expected_gain = (
                    prior["mean"]
                    / _CHANNEL_MULTIPLIERS["sms"]
                    * float(cell.mean_arpu)
                    * max(scalable_n, 0)
                    - pilot_n * channels["sms"]["cost_per_contact"]
                )
                # A modest exploration bonus prevents sparse history from
                # excluding every uncertain but potentially valuable tariff.
                exploration = (
                    prior["sd"]
                    / _CHANNEL_MULTIPLIERS["sms"]
                    * float(cell.mean_arpu)
                    * max(scalable_n, 0)
                )
                choices.append((expected_gain + 0.35 * exploration, target, prior))

            if not choices:
                continue
            _, target, prior = max(choices, key=lambda choice: choice[0])
            candidates.append(
                {
                    "source": source,
                    "segment": segment,
                    "target": target,
                    "n": audience_n,
                    "mean_arpu": float(cell.mean_arpu),
                    "prior_mean": prior["mean"],
                    "prior_sd": prior["sd"],
                    "history_n": prior["history_n"],
                    "observations": [],
                    "pilot_n": 0,
                }
            )

        # One target hypothesis per disjoint current-tariff/ARPU cell avoids
        # spending multiple pilots on nearly identical audiences.
        candidates.sort(
            key=lambda c: (
                (c["prior_mean"] + 0.35 * c["prior_sd"])
                * c["mean_arpu"]
                * max(min(c["n"], _MAX_CAMPAIGN_SIZE) - _PILOT_SIZE, 0)
            ),
            reverse=True,
        )

        available_pilots = min(_MAX_PILOTS, int(env.pilots_left))
        untested = candidates.copy()
        tested = []
        while untested and len(tested) < available_pilots:
            # Re-rank after every observation. The upper confidence bound gives
            # uncertain high-value hypotheses a chance while deprioritizing
            # candidates whose plausible upside is already small.
            def acquisition(candidate):
                mean, sd = self._posterior(
                    candidate["prior_mean"], candidate["prior_sd"], candidate["observations"]
                )
                upper = max(mean + 0.5 * sd, 0.0)
                pilot_n = min(_PILOT_SIZE, candidate["n"])
                remaining_n = min(candidate["n"], _MAX_CAMPAIGN_SIZE) - pilot_n
                gross = (
                    upper
                    / _CHANNEL_MULTIPLIERS["sms"]
                    * candidate["mean_arpu"]
                    * max(remaining_n, 0)
                )
                pilot_cost = pilot_n * channels["sms"]["cost_per_contact"]
                return gross - pilot_cost

            candidate = max(untested, key=acquisition)
            if acquisition(candidate) <= 0:
                break
            untested.remove(candidate)
            pilot_n = min(_PILOT_SIZE, candidate["n"])
            if env.remaining_contacts < pilot_n:
                break
            if pilot_n * channels["sms"]["cost_per_contact"] > env.remaining_budget:
                break

            try:
                result = env.run_pilot(
                    target_tariff=candidate["target"],
                    channel="sms",
                    n_customers=pilot_n,
                    filter_arpu_segment=candidate["segment"],
                    filter_current_tariff=candidate["source"],
                )
            except (RuntimeError, ValueError):
                continue

            candidate["pilot_n"] = int(result["n_customers"])
            candidate["observations"].append(
                (float(result["observed_lift_ratio"]), candidate["pilot_n"], "sms")
            )
            tested.append(candidate)

        if not tested:
            return []

        # Build conservative campaign choices. The pilot customers are already
        # contacted, so only the remaining audience contributes incremental
        # lift while every final campaign contact still incurs its channel cost.
        campaign_options = []
        for candidate in tested:
            posterior_mean, posterior_sd = self._posterior(
                candidate["prior_mean"], candidate["prior_sd"], candidate["observations"]
            )
            lower_sms = posterior_mean - _ONE_SIDED_Z * posterior_sd
            audience_n = min(candidate["n"], _MAX_CAMPAIGN_SIZE)
            untouched_n = max(audience_n - candidate["pilot_n"], 0)
            if untouched_n == 0:
                continue

            best_option = None
            for channel, channel_info in channels.items():
                multiplier = float(
                    channel_info.get("conversion_multiplier", _CHANNEL_MULTIPLIERS.get(channel, 0))
                )
                if multiplier <= 0:
                    continue
                cost_per_contact = float(channel_info["cost_per_contact"])
                expected_lift = (
                    posterior_mean
                    / _CHANNEL_MULTIPLIERS["sms"]
                    * multiplier
                    * candidate["mean_arpu"]
                    * untouched_n
                )
                conservative_lift = (
                    lower_sms
                    / _CHANNEL_MULTIPLIERS["sms"]
                    * multiplier
                    * candidate["mean_arpu"]
                    * untouched_n
                )
                campaign_cost = audience_n * cost_per_contact
                expected_net = expected_lift - campaign_cost
                conservative_net = conservative_lift - campaign_cost
                option = {
                    "channel": channel,
                    "expected_net": expected_net,
                    "conservative_net": conservative_net,
                    "cost_per_contact": cost_per_contact,
                }
                if best_option is None or option["expected_net"] > best_option["expected_net"]:
                    best_option = option

            if best_option is None or best_option["expected_net"] <= 0:
                continue
            campaign_options.append((candidate, best_option))

        # Prefer candidates with positive lower-confidence-bound net value.
        # If none survive (possible with a very noisy mock seed), keep the best
        # positive-posterior option as a guarded fallback so the contract still
        # returns a valid campaign.
        robust = [item for item in campaign_options if item[1]["conservative_net"] > 0]
        selected_pool = robust or campaign_options
        selected_pool.sort(
            key=lambda item: (
                item[1]["conservative_net"] if robust else item[1]["expected_net"]
            ),
            reverse=True,
        )

        campaigns = []
        remaining_contacts = int(env.remaining_contacts)
        remaining_budget = float(env.remaining_budget)
        for candidate, option in selected_pool:
            if len(campaigns) >= _MAX_CAMPAIGNS or remaining_contacts <= 0:
                break
            audience_n = min(candidate["n"], _MAX_CAMPAIGN_SIZE, remaining_contacts)
            cost = audience_n * option["cost_per_contact"]
            if option["cost_per_contact"] > 0:
                audience_n = min(audience_n, int(remaining_budget // option["cost_per_contact"]))
                cost = audience_n * option["cost_per_contact"]
            if audience_n <= candidate["pilot_n"]:
                continue
            if audience_n <= 0:
                continue

            campaigns.append(
                {
                    "campaign_name": f"{candidate['source']}_{candidate['segment']}_{candidate['target']}",
                    "filter_arpu_segment": candidate["segment"],
                    "filter_current_tariff": candidate["source"],
                    "target_tariff": candidate["target"],
                    "channel": option["channel"],
                }
            )
            remaining_contacts -= audience_n
            remaining_budget -= cost

        return campaigns
