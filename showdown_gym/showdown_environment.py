import os
import time
from typing import Any, Dict, Iterable, Optional

import numpy as np
from poke_env import (
    AccountConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.player import Player

from showdown_gym.base_environment import BaseShowdownEnv

# ----------------------
# Type chart + helpers
# ----------------------
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

def type_effectiveness_single(atk_type, defender_types: Iterable) -> float:
    """Effectiveness of a single attack type against up to two defender types."""
    if not atk_type or not defender_types:
        return 1.0
    row = TYPE_CHART.get(_type_name(atk_type), {})
    mult = 1.0
    for df in defender_types:
        mult *= row.get(_type_name(df), 1.0)
    return mult  # ∈ {0, 0.25, 0.5, 1, 2, 4}

def normalized_matchup_beststab(attacker_types, defender_types) -> float:
    """Best single-type (STAB-like) matchup proxy ∈ [0,1]."""
    if not attacker_types or not defender_types:
        return 0.5
    best = 0.0
    for atk in attacker_types:
        best = max(best, type_effectiveness_single(atk, defender_types))
    return min(best / 4.0, 1.0)

def is_stab(move_type, my_types) -> bool:
    if not move_type or not my_types:
        return False
    mt = _type_name(move_type)
    return any(mt == _type_name(t) for t in my_types)

def hazard_scalar(side_conditions) -> float:
    """Tiny compressed hazard scalar ∈ [0,1] (roughly).
    SR=0.3, Spikes=0.15/layer, TSpikes=0.2/layer, Web=0.25.
    """
    weights = {"stealthrock": 0.3, "spikes": 0.15, "toxicspikes": 0.2, "stickyweb": 0.25}
    total = 0.0
    for sc_key, sc_val in (side_conditions or {}).items():
        name = str(getattr(sc_key, "name", sc_key)).lower()
        if name in weights:
            try:
                layers = int(sc_val)
            except Exception:
                layers = 1 if sc_val else 0
            total += weights[name] * layers
    return float(np.clip(total, 0.0, 1.0))

# =======================
# Focused-30 Environment
# =======================
class ShowdownEnvironment(BaseShowdownEnv):
    """
    Observation (30 dims exactly):
      team_hp[6], opp_hp[6],                               # 12
      matchup_my_to_opp, matchup_opp_to_my,               # 2  -> 14
      move_eff[4], move_immune[4],                        # 8  -> 22
      best_proxy,                                         # 1  -> 23
      speed_edge, speed_ratio,                            # 2  -> 25
      dmg_dealt_last, dmg_taken_last,                     # 2  -> 27
      opp_low_hp_flag,                                    # 1  -> 28
      my_haz_scalar, opp_haz_scalar                       # 2  -> 30
    (No bias term to keep 30.)
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
    
    def _get_action_size(self) -> int | None:
        """
        None just uses the default number of actions as laid out in process_action - 26 actions.

        This defines the size of the action space for the agent - e.g. the output of the RL agent.

        This should return the number of actions you wish to use if not using the default action scheme.
        """
        return 10  # Return None if action size is default

    def process_action(self, action: np.int64) -> np.int64:
        """
        Returns the np.int64 relative to the given action.

        The action mapping is as follows:
        action = -2: default
        action = -1: forfeit
        0 <= action <= 5: switch
        6 <= action <= 9: move
        10 <= action <= 13: move and mega evolve
        14 <= action <= 17: move and z-move
        18 <= action <= 21: move and dynamax
        22 <= action <= 25: move and terastallize

        :param action: The action to take.
        :type action: int64

        :return: The battle order ID for the given action in context of the current battle.
        :rtype: np.Int64
        """
        #print(action)
        return action

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()
        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won
        return info

    # ---------- helpers ----------
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
    def _team_hp_list(team_dict):
        hp = []
        for mon in team_dict.values():
            hp.append(float(getattr(mon, "current_hp_fraction", 1.0) or 0.0) if mon else 1.0)
        hp = hp[:6]
        if len(hp) < 6:
            hp = hp + [1.0] * (6 - len(hp))
        return hp

    @staticmethod
    def _team_hp_sum(team_dict):
        return sum(float(getattr(mon, "current_hp_fraction", 0.0) or 0.0) for mon in team_dict.values())

    @staticmethod
    def _best_move_features(avail_moves, opp_types, my_types):
        """Returns (eff[4], immune[4], best_proxy[0..1]).
        best_proxy incorporates STAB and is normalized by 150*4*1.5=900.
        """
        effs, immunes = [], []
        best_proxy = 0.0
        for move in (avail_moves or [])[:4]:
            bp = float(getattr(move, "base_power", 0.0) or 0.0)
            mtype = getattr(move, "type", None)
            eff_raw = type_effectiveness_single(mtype, opp_types) if opp_types else 1.0
            stab = 1.5 if is_stab(mtype, my_types) else 1.0
            effs.append(min(eff_raw / 4.0, 1.0))
            immunes.append(1.0 if eff_raw == 0.0 else 0.0)
            best_proxy = max(best_proxy, (bp * eff_raw * stab) / 900.0)
        while len(effs) < 4:
            effs.append(0.0); immunes.append(0.0)
        return effs, immunes, max(0.0, min(best_proxy, 1.0))

    # ---------- observation ----------
    def _observation_size(self) -> int:
        return 30

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        team_hp = self._team_hp_list(battle.team)          # 6
        opp_hp  = self._team_hp_list(battle.opponent_team) # 6

        my_act  = getattr(battle, "active_pokemon", None)
        opp_act = getattr(battle, "opponent_active_pokemon", None)
        my_types  = getattr(my_act, "types", []) if my_act else []
        opp_types = getattr(opp_act, "types", []) if opp_act else []

        matchup_my_to_opp = normalized_matchup_beststab(my_types, opp_types)
        matchup_opp_to_my = normalized_matchup_beststab(opp_types, my_types)

        effs, immunes, best_proxy = self._best_move_features(
            getattr(battle, "available_moves", []) or [], opp_types, my_types
        )

        # damage last step
        try:
            prior = self._get_prior_battle(battle)
        except AttributeError:
            prior = None
        my_now_sum  = self._team_hp_sum(battle.team)
        opp_now_sum = self._team_hp_sum(battle.opponent_team)
        if prior is None:
            dmg_dealt_last = 0.0; dmg_taken_last = 0.0
        else:
            my_prev_sum  = self._team_hp_sum(prior.team)
            opp_prev_sum = self._team_hp_sum(prior.opponent_team)
            dmg_dealt_last = max(0.0, opp_prev_sum - opp_now_sum)
            dmg_taken_last = max(0.0, my_prev_sum  - my_now_sum)

        # speed features
        my_speed  = self._approx_speed(my_act)
        opp_speed = self._approx_speed(opp_act)
        speed_edge  = 1.0 if my_speed > opp_speed else 0.0
        speed_ratio = max(0.0, min(my_speed / max(1e-6, opp_speed), 3.0)) / 3.0

        # low-HP flag
        opp_low_hp_flag = 1.0 if float(getattr(opp_act, "current_hp_fraction", 1.0) or 0.0) < 0.25 else 0.0

        # hazards compressed
        my_haz  = hazard_scalar(getattr(battle, "side_conditions", {}))
        opp_haz = hazard_scalar(getattr(battle, "opponent_side_conditions", {}))

        obs = (
            team_hp + opp_hp +
            [matchup_my_to_opp, matchup_opp_to_my] +
            effs + immunes +
            [best_proxy] +
            [speed_edge, speed_ratio] +
            [dmg_dealt_last, dmg_taken_last] +
            [opp_low_hp_flag] +
            [my_haz, opp_haz]
        )
        vec = np.array(obs, dtype=np.float32)
        # safety
        if vec.size != self._observation_size():
            if vec.size < self._observation_size():
                vec = np.pad(vec, (0, self._observation_size() - vec.size))
            else:
                vec = vec[: self._observation_size()]
        return vec

    def calc_reward(self, battle: AbstractBattle) -> float:
        """Purely positive shaping:
           + damage dealt (scaled & clipped)
           + took the highest-efficiency move (ties OK; tiny extra if STAB)
           + switched out of low-eff and improved matchup
           + enemy KO
           + terminal bonus for # of my Pokémon alive (+ small HP-advantage)
        """
        try:
            prior = self._get_prior_battle(battle)
        except AttributeError:
            prior = None

        # short-hands
        def team_hp_sum(team_dict):
            return sum(float(getattr(mon, "current_hp_fraction", 0.0) or 0.0) for mon in team_dict.values())

        def faint_count(team_dict):
            return sum(
                1 for mon in team_dict.values()
                if mon and (getattr(mon, "fainted", False) or getattr(mon, "current_hp", 0) <= 0)
            )

        def alive_count(team_dict):
            return sum(
                1 for mon in team_dict.values()
                if mon and not getattr(mon, "fainted", False) and getattr(mon, "current_hp", 0) > 0
            )

        def best_eff_against(opp_types, moves_list, my_types) -> float:
            best = 0.0
            for mv in (moves_list or [])[:4]:
                mtype = getattr(mv, "type", None)
                eff = type_effectiveness_single(mtype, opp_types) if opp_types else 1.0
                if is_stab(mtype, my_types):
                    eff *= 1.01  # nudge ties toward STAB
                best = max(best, eff)
            return best  # raw (0..4*1.01)

        def last_my_move(b: AbstractBattle):
            # Return (type, is_stab) if we can infer; else (None, False)
            try:
                mv = getattr(b, "last_move", None)
                if mv and getattr(mv, "type", None):
                    ap = getattr(b, "active_pokemon", None)
                    my_types = getattr(ap, "types", []) if ap else []
                    return mv.type, is_stab(mv.type, my_types)
            except Exception:
                pass
            try:
                ap = getattr(b, "active_pokemon", None)
                mv = getattr(ap, "last_move_used", None)
                if mv and getattr(mv, "type", None):
                    my_types = getattr(ap, "types", []) if ap else []
                    return mv.type, is_stab(mv.type, my_types)
            except Exception:
                pass
            return None, False

        # current snapshot
        my_now  = team_hp_sum(battle.team)
        opp_now = team_hp_sum(battle.opponent_team)

        if prior is None:
            return 0.0  # start from zero, positive-only

        my_prev  = team_hp_sum(prior.team)
        opp_prev = team_hp_sum(prior.opponent_team)

        reward = 0.0

        # (1) Damage dealt (dense, positive, stabilized)
        dmg_dealt = max(0.0, opp_prev - opp_now)       # ∈ [0..6]
        reward += 0.5 * float(min(dmg_dealt, 1.0))     # cap per-step contribution at 0.5

        # (2) KO bonus (spiky, positive)
        opp_faints_prev = faint_count(prior.opponent_team)
        opp_faints_now  = faint_count(battle.opponent_team)
        if opp_faints_now > opp_faints_prev:
            reward += 2.0 * (opp_faints_now - opp_faints_prev)

        # (3) Highest-eff move picked (ties OK; tiny STAB boost; non-immune nudge)
        prev_opp_act = getattr(prior, "opponent_active_pokemon", None)
        prev_my_act  = getattr(prior, "active_pokemon", None)
        prev_opp_types = getattr(prev_opp_act, "types", []) if prev_opp_act else []
        prev_my_types  = getattr(prev_my_act, "types", []) if prev_my_act else []
        prev_moves = getattr(prior, "available_moves", []) or []
        best_eff_prev = best_eff_against(prev_opp_types, prev_moves, prev_my_types)  # raw

        used_type, used_was_stab = last_my_move(battle)
        if used_type is not None and prev_opp_types:
            used_eff = type_effectiveness_single(used_type, prev_opp_types)
            if used_eff >= best_eff_prev - 1e-6:
                reward += 0.2
                if used_was_stab:
                    reward += 0.05
            # tiny positive for not hitting an immunity
            if used_eff > 0.0:
                reward += 0.05

        # (4) Smart switch when efficiencies were low (slightly gentler threshold)
        now_my_act  = getattr(battle, "active_pokemon", None)
        now_opp_act = getattr(battle, "opponent_active_pokemon", None)
        switched = (prev_my_act is not None) and (now_my_act is not None) and (prev_my_act is not now_my_act)
        if switched:
            now_my_types  = getattr(now_my_act, "types", []) if now_my_act else []
            now_opp_types = getattr(now_opp_act, "types", []) if now_opp_act else []
            prev_best_norm = min(best_eff_prev / 4.0, 1.0) if best_eff_prev > 0 else 0.0
            now_matchup = normalized_matchup_beststab(now_my_types, now_opp_types)
            if prev_best_norm < 0.5 and now_matchup > prev_best_norm + 0.20:
                reward += 0.3

        # (5) Terminal positive bonuses
        if battle.finished:
            # alive bonus
            my_alive = sum(
                1 for m in battle.team.values()
                if m and not getattr(m, "fainted", False) and getattr(m, "current_hp", 0) > 0
            )
            reward += 0.5 * my_alive  # up to +3.0

            # slight HP-advantage bonus (positive-only)
            hp_margin = max(0.0, my_now - opp_now)     # only reward if ahead
            reward += 0.2 * float(hp_margin)           # up to +1.2 if you finish very healthy

        # strictly non-negative and stable decimals
        return max(0.0, round(reward, 3))


########################################
# DO NOT EDIT THE CODE BELOW THIS LINE #
########################################

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
        unique_id = time.strftime("%H%M%S")

        opponent_account = "ot" if not evaluation else "oe"
        opponent_account = f"{opponent_account}_{unique_id}"

        opponent_configuration = AccountConfiguration(opponent_account, None)
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "random":
            opponent = RandomPlayer(account_configuration=opponent_configuration)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "t1" if not evaluation else "e1"
        account_name_two: str = "t2" if not evaluation else "e2"

        account_name_one = f"{account_name_one}_{unique_id}"
        account_name_two = f"{account_name_two}_{unique_id}"

        team = self._load_team(team_type)
        battle_format = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_format,
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
