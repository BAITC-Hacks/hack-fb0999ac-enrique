"""Enrique-agent: budget-aware tariff campaign agent.

History ranks hypotheses, while two pilot stages confirm the hypotheses before
the agent allocates the remaining contacts and communication budget.
"""

from pathlib import Path

import numpy as np
import pandas as pd


_SMS_SCALE = 0.65
_PILOT_STD_PER_CUSTOMER = 0.804
_CONFIDENCE_Z = 1.28  # One-sided 90% lower confidence bound.
_MAX_CAMPAIGNS = 10
_MAX_CAMPAIGN_SIZE = 5_000
_MAX_PILOTS = 20
_FIRST_STAGE_PILOTS = 12
_FIRST_STAGE_SIZE = 100
_SECOND_STAGE_PILOTS = 4
_SECOND_STAGE_SIZE = 200


class Agent:
    """Select and validate tariff campaign hypotheses within env limits."""

    def _historical_priors(self, tariff_codes):
        """Build weak SMS-scale uplift priors from historical transitions."""
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
            history = history.loc[
                history["tariff_plan_code_from"] != history["tariff_plan_code_to"]
            ].copy()

            grouped = history.groupby(
                ["tariff_plan_code_from", "arpu_segment", "tariff_plan_code_to"],
                observed=True,
            )["relative_change"].agg(["mean", "std", "count"])
            source_counts = history.groupby(
                ["tariff_plan_code_from", "arpu_segment"], observed=True
            ).size()

            priors = {}
            for (source, segment, target), row in grouped.iterrows():
                if source not in tariff_codes or target not in tariff_codes:
                    continue
                source_count = int(source_counts.get((source, segment), 0))
                if source_count == 0:
                    continue

                # A historical transition is not a randomized campaign result.
                # Its destination frequency therefore only weakly scales the prior.
                transition_share = float(row["count"]) / source_count
                mean = float(row["mean"]) * transition_share * _SMS_SCALE
                raw_sd = float(row["std"]) if pd.notna(row["std"]) else 0.0
                sd = max(0.12, min(0.35, raw_sd / np.sqrt(max(row["count"], 1))))
                priors[(source, str(segment), target)] = {
                    "mean": mean,
                    "sd": sd * _SMS_SCALE,
                    "history_n": int(row["count"]),
                }
            return priors
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            return {}

    @staticmethod
    def _posterior(prior_mean, prior_sd, observations, channels):
        """Combine a weak prior with noisy pilot ratios on the SMS scale."""
        precision = 1.0 / max(prior_sd**2, 1e-8)
        weighted_sum = prior_mean * precision

        for observed_ratio, sample_size, channel in observations:
            multiplier = float(channels[channel]["conversion_multiplier"])
            if multiplier <= 0:
                continue
            normalized_ratio = observed_ratio * _SMS_SCALE / multiplier
            noise_sd = _PILOT_STD_PER_CUSTOMER / np.sqrt(max(sample_size, 1))
            normalized_noise_sd = noise_sd * _SMS_SCALE / multiplier
            observation_precision = 1.0 / max(normalized_noise_sd**2, 1e-8)
            precision += observation_precision
            weighted_sum += normalized_ratio * observation_precision

        return weighted_sum / precision, np.sqrt(1.0 / precision)

    @staticmethod
    def _audience_cells(profile):
        return (
            profile.groupby(["current_tariff", "arpu_segment"], observed=True)
            .agg(n=("ID_NUMBER", "size"), mean_arpu=("predicted_arpu", "mean"))
            .reset_index()
        )

    @staticmethod
    def _estimated_unique_pilot_contacts(candidate):
        """Estimate unique people in repeated random pilot samples of one cell."""
        audience_n = max(candidate["n"], 1)
        not_contacted_probability = 1.0
        for _, sample_size, _ in candidate["observations"]:
            not_contacted_probability *= max(1.0 - sample_size / audience_n, 0.0)
        return audience_n * (1.0 - not_contacted_probability)

    def _best_channel_option(self, candidate, channels, budget, contacts, risk_z, objective="expected"):
        """Evaluate every channel on the capacity available at this decision."""
        mean_sms, sd_sms = self._posterior(
            candidate["prior_mean"], candidate["prior_sd"], candidate["observations"], channels
        )
        lower_sms = mean_sms - risk_z * sd_sms
        max_segment_n = min(candidate["n"], _MAX_CAMPAIGN_SIZE)
        estimated_pilot_contacts = self._estimated_unique_pilot_contacts(candidate)
        unpiloted_share = max(1.0 - estimated_pilot_contacts / candidate["n"], 0.0)

        options = []
        for channel, channel_info in channels.items():
            multiplier = float(channel_info["conversion_multiplier"])
            cost_per_contact = float(channel_info["cost_per_contact"])
            budget_cap = max_segment_n
            if cost_per_contact > 0:
                budget_cap = int(max(budget, 0) // cost_per_contact)
            campaign_contacts = min(max_segment_n, int(contacts), budget_cap)
            if campaign_contacts <= 0 or multiplier <= 0:
                continue

            # Final campaigns include their pilot audience because the public
            # contract has filters but no exclusion list. Repeated contacts cost
            # money, while the expected uplift counts only once per customer.
            incremental_customers = campaign_contacts * unpiloted_share
            expected_lift = (
                mean_sms / _SMS_SCALE * multiplier * candidate["mean_arpu"] * incremental_customers
            )
            lower_lift = (
                lower_sms / _SMS_SCALE * multiplier * candidate["mean_arpu"] * incremental_customers
            )
            cost = campaign_contacts * cost_per_contact
            options.append(
                {
                    "channel": channel,
                    "contacts": campaign_contacts,
                    "cost": cost,
                    "expected_net": expected_lift - cost,
                    "conservative_net": lower_lift - cost,
                    "posterior_mean_sms": mean_sms,
                    "posterior_sd_sms": sd_sms,
                }
            )

        if not options:
            return None
        return max(options, key=lambda option: option[f"{objective}_net"])

    def _run_pilot(self, env, candidate, sample_size):
        """Run one affordable SMS pilot and retain its observed result."""
        cost_per_contact = float(env.channels["sms"]["cost_per_contact"])
        affordable_by_budget = int(env.remaining_budget // cost_per_contact)
        actual_size = min(sample_size, candidate["n"], env.remaining_contacts, affordable_by_budget)
        if actual_size < 10 or env.pilots_left <= 0:
            return False
        try:
            result = env.run_pilot(
                target_tariff=candidate["target"],
                channel="sms",
                n_customers=actual_size,
                filter_arpu_segment=candidate["segment"],
                filter_current_tariff=candidate["source"],
            )
        except (RuntimeError, ValueError):
            return False

        candidate["observations"].append(
            (float(result["observed_lift_ratio"]), int(result["n_customers"]), "sms")
        )
        return True

    def act(self, env):
        profile = env.customer_profile
        channels = env.channels
        tariff_codes = set(env.tariffs["tariff_plan_code"].astype(str))
        prices = env.tariffs.set_index("tariff_plan_code")["price_tariff"].to_dict()
        priors = self._historical_priors(tariff_codes)

        candidates = []
        for cell in self._audience_cells(profile).itertuples(index=False):
            source, segment, audience_n = str(cell.current_tariff), str(cell.arpu_segment), int(cell.n)
            if audience_n < _FIRST_STAGE_SIZE or source not in prices:
                continue

            choices = []
            for target in sorted(tariff_codes):
                if target == source or prices[target] < prices[source]:
                    continue
                prior = priors.get((source, segment, target))
                if prior is None or prior["mean"] <= 0:
                    continue

                scale = min(audience_n, _MAX_CAMPAIGN_SIZE)
                rank = (prior["mean"] + 0.35 * prior["sd"]) * float(cell.mean_arpu) * scale
                choices.append((rank, target, prior))

            if choices:
                _, target, prior = max(choices, key=lambda item: item[0])
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
                    }
                )

        candidates.sort(
            key=lambda candidate: (
                (candidate["prior_mean"] + 0.35 * candidate["prior_sd"])
                * candidate["mean_arpu"]
                * min(candidate["n"], _MAX_CAMPAIGN_SIZE)
            ),
            reverse=True,
        )

        # Stage 1: broad screening. It leaves budget and pilot capacity for
        # confirmation instead of treating the first noisy observation as proof.
        screened = []
        screening_limit = min(_FIRST_STAGE_PILOTS, _MAX_PILOTS, env.pilots_left)
        for candidate in candidates[:screening_limit]:
            if self._run_pilot(env, candidate, _FIRST_STAGE_SIZE):
                screened.append(candidate)

        if not screened:
            return []

        # Stage 2: repeat the most valuable or ambiguous candidates at a larger
        # sample. This includes candidates with a plausible upside even if the
        # first pilot was not decisively positive.
        confirmation_capacity = min(
            _SECOND_STAGE_PILOTS,
            _MAX_PILOTS - len(screened),
            env.pilots_left,
        )

        def confirmation_score(candidate):
            option = self._best_channel_option(
                candidate,
                channels,
                env.remaining_budget,
                env.remaining_contacts,
                risk_z=0.0,
            )
            if option is None:
                return float("-inf")
            upper_sms = option["posterior_mean_sms"] + option["posterior_sd_sms"]
            upper_value = (
                upper_sms / _SMS_SCALE * candidate["mean_arpu"] * min(candidate["n"], _MAX_CAMPAIGN_SIZE)
            )
            return option["expected_net"] + 0.25 * max(upper_value - option["expected_net"], 0.0)

        confirmation_candidates = sorted(
            screened, key=confirmation_score, reverse=True
        )[:confirmation_capacity]
        for candidate in confirmation_candidates:
            self._run_pilot(env, candidate, _SECOND_STAGE_SIZE)

        # Allocate final campaigns greedily. Before every selection all channel
        # options are recalculated against the real remaining budget and reach.
        remaining_budget = float(env.remaining_budget)
        remaining_contacts = int(env.remaining_contacts)
        pending = screened.copy()
        campaigns = []

        while pending and len(campaigns) < _MAX_CAMPAIGNS and remaining_contacts > 0:
            evaluated = []
            for candidate in pending:
                expected_option = self._best_channel_option(
                    candidate,
                    channels,
                    remaining_budget,
                    remaining_contacts,
                    risk_z=_CONFIDENCE_Z,
                )
                conservative_option = self._best_channel_option(
                    candidate,
                    channels,
                    remaining_budget,
                    remaining_contacts,
                    risk_z=_CONFIDENCE_Z,
                    objective="conservative",
                )
                if expected_option is not None:
                    evaluated.append((candidate, expected_option, conservative_option))

            robust = [
                (candidate, conservative_option)
                for candidate, _, conservative_option in evaluated
                if conservative_option is not None and conservative_option["conservative_net"] > 0
            ]
            if robust:
                candidate, option = max(robust, key=lambda item: item[1]["conservative_net"])
            else:
                expected_positive = [
                    (candidate, expected_option)
                    for candidate, expected_option, _ in evaluated
                    if expected_option["expected_net"] > 0
                ]
                if not expected_positive:
                    break
                candidate, option = max(expected_positive, key=lambda item: item[1]["expected_net"])

            campaigns.append(
                {
                    "campaign_name": f"{candidate['source']}_{candidate['segment']}_{candidate['target']}",
                    "filter_arpu_segment": candidate["segment"],
                    "filter_current_tariff": candidate["source"],
                    "target_tariff": candidate["target"],
                    "channel": option["channel"],
                }
            )
            pending.remove(candidate)
            remaining_budget -= option["cost"]
            remaining_contacts -= option["contacts"]

        return campaigns
