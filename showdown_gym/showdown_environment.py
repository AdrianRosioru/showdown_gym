import os
from typing import Any, Dict

import numpy as np
from poke_env import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.player import Player

from showdown_gym.base_environment import BaseShowdownEnv

# ----------------------
# Type chart + utilities
# ----------------------
# All keys are lowercase strings
TYPE_CHART = {
    "normal": {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fire":   {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 2.0, "bug": 2.0, "rock": 0.5, "dragon": 0.5, "steel": 2.0},
    "water":  {"fire": 2.0, "water": 0.5, "grass": 0.5, "ground": 2.0, "rock": 2.0, "dragon": 0.5},
    "electric":{"water": 2.0, "electric": 0.5, "grass": 0.5, "ground": 0.0, "flying": 2.0, "dragon": 0.5},
    "grass":  {"fire": 0.5, "water": 2.0, "grass": 0.5, "poison": 0.5, "ground": 2.0, "flying": 0.5, "bug": 0.5, "rock": 2.0, "dragon": 0.5, "steel": 0.5},
    "ice":    {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 0.5, "ground": 2.0, "flying": 2.0, "dragon": 2.0, "steel": 0.5},
    "fighting":{"normal": 2.0, "ice": 2.0, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2.0, "ghost": 0.0, "dark": 2.0, "steel": 2.0, "fairy": 0.5},
    "poison": {"grass": 2.0, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0.0, "fairy": 2.0},
    "ground": {"fire": 2.0, "electric": 2.0, "grass": 0.5, "poison": 2.0, "flying": 0.0, "bug": 0.5, "rock": 2.0, "steel": 2.0},
    "flying": {"electric": 0.5, "grass": 2.0, "fighting": 2.0, "bug": 2.0, "rock": 0.5, "steel": 0.5},
    "psychic":{"fighting": 2.0, "poison": 2.0, "psychic": 0.5, "dark": 0.0, "steel": 0.5},
    "bug":    {"fire": 0.5, "grass": 2.0, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2.0, "ghost": 0.5, "dark": 2.0, "steel": 0.5, "fairy": 0.5},
    "rock":   {"fire": 2.0, "ice": 2.0, "fighting": 0.5, "ground": 0.5, "flying": 2.0, "bug": 2.0, "steel": 0.5},
    "ghost":  {"normal": 0.0, "psychic": 2.0, "ghost": 2.0, "dark": 0.5},
    "dragon": {"dragon": 2.0, "steel": 0.5, "fairy": 0.0},
    "dark":   {"fighting": 0.5, "psychic": 2.0, "ghost": 2.0, "dark": 0.5, "fairy": 0.5},
    "steel":  {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2.0, "rock": 2.0, "fairy": 2.0, "steel": 0.5},
    "fairy":  {"fire": 0.5, "fighting": 2.0, "poison": 0.5, "dragon": 2.0, "dark": 2.0, "steel": 0.5},
}
def _type_name(t) -> str:
    return (t.name if hasattr(t, "name") else str(t)).lower()

def offensive_multiplier(attacker_types, defender_types) -> float:
    mult = 1.0
    if not attacker_types or not defender_types:
        return mult
    for atk in attacker_types:
        row = TYPE_CHART.get(_type_name(atk), {})
        for df in defender_types:
            mult *= row.get(_type_name(df), 1.0)
    return mult

def normalized_offensive_multiplier(attacker_types, defender_types) -> float:
    # normalize by max 4x
    return offensive_multiplier(attacker_types, defender_types) / 4.0


# =======================
# Simplified Environment
# =======================
class ShowdownEnvironment(BaseShowdownEnv):
    """
    Tiny high-signal observation (12 dims) and spiky reward for truly good outcomes.
    Observation:
        [ my_hp, opp_hp, my_rem, opp_rem,
          my_vs_opp, opp_vs_my,
          best_eff, best_bp, best_proxy,
          speed_edge, opp_low_hp_flag, bias ]
    """

    # ---------- init / info ----------
    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()
        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won
        return info

    # ---------- tiny helpers ----------
    @staticmethod
    def _stage_mult(stage: int) -> float:
        s = int(stage)
        return (2 + s) / 2.0 if s >= 0 else 2.0 / (2 - s)

    def _approx_speed(self, mon):
        if not mon:
            return 1.0
        base_stats = getattr(mon, "base_stats", {}) or {}
        base_spe = float(base_stats.get("spe", 50.0))
        boosts = getattr(mon, "boosts", {}) or {}
        mult = self._stage_mult(int(boosts.get("spe", 0)))
        return base_spe * mult

    @staticmethod
    def _remaining(team_dict) -> int:
        cnt = 0
        for mon in team_dict.values():
            if mon and not getattr(mon, "fainted", False) and getattr(mon, "current_hp", 0) > 0:
                cnt += 1
        return cnt

    @staticmethod
    def _best_move_stats(avail_moves, opp_types):
        """Return best effectiveness, best base power, and a crude damage proxy in [0,1]."""
        best_eff = 0.0
        best_bp = 0.0
        best_proxy = 0.0
        for move in (avail_moves or []):
            bp = float(getattr(move, "base_power", 0.0) or 0.0)
            mtype = getattr(move, "type", None)
            eff = offensive_multiplier([mtype] if mtype else [], opp_types) if opp_types else 1.0
            # proxy: normalize by 150*4=600 (randbats-friendly)
            proxy = (bp * eff) / 600.0
            if eff > best_eff:
                best_eff = eff
            if bp > best_bp:
                best_bp = bp
            if proxy > best_proxy:
                best_proxy = proxy
        # normalize for obs
        best_eff_norm = min(best_eff / 4.0, 1.0)
        best_bp_norm = min(best_bp / 150.0, 1.0)
        best_proxy = max(0.0, min(best_proxy, 1.0))
        return best_eff_norm, best_bp_norm, best_proxy

    # ---------- observation ----------
    def _observation_size(self) -> int:
        return 12

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        MAX_TEAM = 6

        my_active  = getattr(battle, "active_pokemon", None)
        opp_active = getattr(battle, "opponent_active_pokemon", None)

        my_hp  = float(getattr(my_active, "current_hp_fraction", 1.0) or 0.0)
        opp_hp = float(getattr(opp_active, "current_hp_fraction", 1.0) or 0.0)

        my_rem  = self._remaining(battle.team) / MAX_TEAM
        opp_rem = self._remaining(battle.opponent_team) / MAX_TEAM

        my_vs_opp = normalized_offensive_multiplier(
            getattr(my_active, "types", []) if my_active else [],
            getattr(opp_active, "types", []) if opp_active else []
        )
        opp_vs_my = normalized_offensive_multiplier(
            getattr(opp_active, "types", []) if opp_active else [],
            getattr(my_active, "types", []) if my_active else []
        )

        best_eff, best_bp, best_proxy = self._best_move_stats(
            getattr(battle, "available_moves", []) or [],
            getattr(opp_active, "types", []) if opp_active else []
        )

        speed_edge = 1.0 if self._approx_speed(my_active) > self._approx_speed(opp_active) else 0.0
        opp_low_hp_flag = 1.0 if opp_hp < 0.25 else 0.0

        obs = [
            my_hp, opp_hp,
            my_rem, opp_rem,
            my_vs_opp, opp_vs_my,
            best_eff, best_bp, best_proxy,
            speed_edge,
            opp_low_hp_flag,
            1.0,  # bias
        ]
        return np.array(obs, dtype=np.float32)

    # ---------- reward (sharp & simple) ----------
    def calc_reward(self, battle: AbstractBattle) -> float:
        prior = self._get_prior_battle(battle)

        def team_hp_sum(team_dict):
            s = 0.0
            for mon in team_dict.values():
                s += float(getattr(mon, "current_hp_fraction", 0.0) or 0.0)
            return s

        # current
        my_now  = team_hp_sum(battle.team)
        opp_now = team_hp_sum(battle.opponent_team)
        my_act  = getattr(battle, "active_pokemon", None)
        opp_act = getattr(battle, "opponent_active_pokemon", None)
        opp_types = getattr(opp_act, "types", []) if opp_act else []
        best_eff_now, best_bp_now, best_proxy_now = self._best_move_stats(
            getattr(battle, "available_moves", []) or [], opp_types
        )
        speed_edge_now = 1.0 if self._approx_speed(my_act) > self._approx_speed(opp_act) else 0.0

        if prior is None:
            return -0.003  # tiny step cost at battle start

        # previous snapshot
        my_prev  = team_hp_sum(prior.team)
        opp_prev = team_hp_sum(prior.opponent_team)
        prev_my_act  = getattr(prior, "active_pokemon", None)
        prev_opp_act = getattr(prior, "opponent_active_pokemon", None)
        prev_opp_types = getattr(prev_opp_act, "types", []) if prev_opp_act else []
        _, _, best_proxy_prev = self._best_move_stats(
            getattr(prior, "available_moves", []) or [], prev_opp_types
        )
        speed_edge_prev = 1.0 if self._approx_speed(prev_my_act) > self._approx_speed(prev_opp_act) else 0.0

        reward = 0.0

        # (1) HP shaping (small)
        reward += (opp_prev - opp_now)  # damage dealt
        reward -= (my_prev - my_now)    # damage taken

        # (2) KO / fainting (spiky)
        def faint_count(team_dict):
            cnt = 0
            for mon in team_dict.values():
                if mon and (getattr(mon, "fainted", False) or getattr(mon, "current_hp", 0) <= 0):
                    cnt += 1
            return cnt

        my_faints_prev  = faint_count(prior.team)
        my_faints_now   = faint_count(battle.team)
        opp_faints_prev = faint_count(prior.opponent_team)
        opp_faints_now  = faint_count(battle.opponent_team)

        reward += 3.0 * max(0, opp_faints_now - opp_faints_prev)  # KO achieved
        reward -= 3.0 * max(0, my_faints_now  - my_faints_prev)   # lost a mon

        # (3) Crossing key HP thresholds for the opponent (first time under 50%, 25%)
        def first_cross(prev_hp_sum, now_hp_sum, threshold_sum):
            return 1.0 if (prev_hp_sum >= threshold_sum and now_hp_sum < threshold_sum) else 0.0

        # Use total opponent HP sum as a proxy for "somebody got into range"
        # (In 6v6, half = 3.0, quarter = 1.5)
        reward += 1.0 * first_cross(opp_prev, opp_now, 3.0)   # under 50% team HP
        reward += 1.5 * first_cross(opp_prev, opp_now, 1.5)   # under 25% team HP

        # (4) Creating a kill threat on the active opponent (proxy >= current opp HP fraction)
        opp_hp_now = float(getattr(opp_act, "current_hp_fraction", 1.0) or 0.0)
        made_threat = (best_proxy_prev < opp_hp_now) and (best_proxy_now >= opp_hp_now)
        if made_threat:
            reward += 0.6

        # (5) Speed edge changes
        if speed_edge_prev == 0.0 and speed_edge_now == 1.0:
            reward += 0.3
        elif speed_edge_prev == 1.0 and speed_edge_now == 0.0:
            reward -= 0.3

        # (6) Terminal outcome
        if battle.finished:
            reward += 8.0 if battle.won else -8.0

        # (7) tiny step cost
        reward -= 0.003

        return float(np.clip(reward, -12.0, 12.0))


# ======================================
# DO NOT EDIT THE CODE BELOW THIS LINE  #
# ======================================
class SingleShowdownWrapper(SingleAgentWrapper):
    """
    A wrapper class for the PokeEnvironment that simplifies the setup of single-agent
    reinforcement learning tasks in a Pokémon battle environment.
    """

    def __init__(
        self,
        team_type: str = "random",
        opponent_type: str = "random",
        evaluation: bool = False,
    ):
        opponent: Player
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer()
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer()
        elif opponent_type == "random":
            opponent = RandomPlayer()
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "train_one" if not evaluation else "eval_one"
        account_name_two: str = "train_two" if not evaluation else "eval_two"

        account_name_one = f"{account_name_one}_{opponent_type}"
        account_name_two = f"{account_name_two}_{opponent_type}"

        team = self._load_team(team_type)

        battle_fomat = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_fomat,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        super().__init__(env=primary_env, opponent=opponent)

    def _load_team(self, team_type: str) -> str | None:
        bot_teams_folders = os.path.join(os.path.dirname(__file__), "teams")

        bot_teams = {}

        for team_file in os.listdir(bot_teams_folders):
            if team_file.endswith(".txt"):
                with open(
                    os.path.join(bot_teams_folders, team_file), "r", encoding="utf-8"
                ) as file:
                    bot_teams[team_file[:-4]] = file.read()

        if team_type in bot_teams:
            return bot_teams[team_type]

        return None
