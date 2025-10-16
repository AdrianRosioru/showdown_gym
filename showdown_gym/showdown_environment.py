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

# =======================
# Focused-30 Environment
# =======================
class ShowdownEnvironment(BaseShowdownEnv):
    """
    Observation: 10-hot action hint only (length 10).
    Exactly one index is 1:
      0..5  -> switch targets
      6..9  -> move indices 0..3
    Heuristic:
      - If our best effectiveness vs opp active < 1x, choose a valid switch that
        maximizes normalized_matchup_beststab.
      - Else choose the available move (0..3) that maximizes (base_power * eff * STAB).
    """
    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
        low_damage_thresh: float = 0.20,  # 20% of a mon
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )
        self.low_damage_thresh = float(low_damage_thresh)
        self._last_action: Optional[int] = None  # for hint-alignment reward
        self._ep_stats = {}          
        self._norm_scale = 10.0   

        self._streak_bonus = {}          # battle_key -> current bonus (starts at 0.1)
        self._streak_bonus_init = 0.1    # reset/start value
        self._streak_bonus_step = 0.1    # increment per consecutive correct
        self._streak_bonus_cap = None    # e.g., 2.0 to cap; None = unlimited
    

    # ---------- action space ----------
    def _get_action_size(self) -> int | None:
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
        return 10

    def process_action(self, action: np.int64) -> np.int64:
        try:
            self._last_action = int(action)
        except Exception:
            self._last_action = None
        return action

    # ---------- helpers (reuses your type/matchup utils) ----------
    @staticmethod
    def _team_hp_sum(team_dict):
        return sum(float(getattr(mon, "current_hp_fraction", 0.0) or 0.0)
                   for mon in team_dict.values())

    def _battle_key(self, battle: AbstractBattle) -> str:
        return getattr(battle, "battle_tag", str(id(battle)))

    def _action_hint_onehot(self, battle: AbstractBattle) -> np.ndarray:
        """
        Minimal, robust heuristic:
        - Score best STAY move by bp * type_effectiveness * STAB * accuracy
        - Consider SWITCH only if best move effectiveness < 1.0 (we're resisted)
            and a living teammate has super-effective coverage (>1.0) vs opp.
        - Among valid switches, prefer high (my->opp) and low (opp->my).
        Maps: 0..5 = switches, 6..9 = moves 0..3.
        """
        onehot = np.zeros(10, dtype=np.float32)

        # ----- tiny local helpers (no external deps besides TYPE_CHART/_type_name) -----
        def eff_vs(atk_type, defender_types) -> float:
            if not atk_type or not defender_types:
                return 1.0
            row = TYPE_CHART.get(_type_name(atk_type), {})
            mult = 1.0
            for df in defender_types:
                mult *= row.get(_type_name(df), 1.0)
            return float(mult)  # can be 0, 0.25, 0.5, 1, 2, 4

        def best_stab_like(attacker_types, defender_types) -> float:
            # proxy "how well does this mon's typing hit them?" in [0,4]
            if not attacker_types or not defender_types:
                return 1.0
            best = 0.0
            for atk in attacker_types:
                best = max(best, eff_vs(atk, defender_types))
            return best

        def is_alive(mon) -> bool:
            return bool(mon and not getattr(mon, "fainted", False) and (getattr(mon, "current_hp", 0) > 0))

        def safe_accuracy(mv) -> float:
            # 1.0 if unknown; normalize 100 -> 1.0
            acc = None
            entry = getattr(mv, "entry", None)
            if isinstance(entry, dict):
                acc = entry.get("accuracy", None)
            if acc in (None, True):
                try:
                    acc = getattr(mv, "accuracy", None)
                except Exception:
                    acc = None
            if acc in (None, True):
                return 1.0
            try:
                acc = float(acc)
                return acc / 100.0 if acc > 1.0 else max(0.0, min(1.0, acc))
            except Exception:
                return 1.0

        # ----- gather state -----
        my_act  = getattr(battle, "active_pokemon", None)
        opp_act = getattr(battle, "opponent_active_pokemon", None)
        my_types  = getattr(my_act, "types", []) if my_act else []
        opp_types = getattr(opp_act, "types", []) if opp_act else []

        # valid switches (0..5)
        team_list = list(battle.team.values())[:6]
        valid_switch_idxs = []
        for i, mon in enumerate(team_list):
            if mon is None or mon is my_act:
                continue
            if is_alive(mon):
                valid_switch_idxs.append(i)

        # valid moves (6..9)
        avail_moves = (getattr(battle, "available_moves", []) or [])[:4]
        valid_moves = []
        for mi, mv in enumerate(avail_moves):
            if not bool(getattr(mv, "disabled", False)):
                valid_moves.append((6 + mi, mv))

        # ----- score "stay" moves -----
        stay_best_a, stay_best_score, stay_best_eff = None, -1.0, 1.0
        for a, mv in valid_moves:
            bp   = float(getattr(mv, "base_power", 0.0) or 0.0)
            mtyp = getattr(mv, "type", None)
            eff  = eff_vs(mtyp, opp_types) if opp_types else 1.0
            if eff == 0.0:
                # never pick into immunity
                continue
            stab = 1.5 if (mtyp and any(_type_name(mtyp) == _type_name(t) for t in (my_types or []))) else 1.0
            acc  = safe_accuracy(mv)
            score = bp * eff * stab * acc
            if score > stay_best_score:
                stay_best_score = score
                stay_best_a = a
                stay_best_eff = eff

        # If no valid moves at all, force a switch if possible (pick safest + best coverage)
        if stay_best_a is None and valid_switch_idxs:
            best_idx, best_tuple = None, (-1e9, -1e9)
            for i in valid_switch_idxs:
                mon = team_list[i]
                new_types = getattr(mon, "types", []) or []
                my_to_opp  = best_stab_like(new_types, opp_types)       # prefer higher
                opp_to_new = best_stab_like(opp_types, new_types)       # prefer lower
                cand = (my_to_opp - opp_to_new, my_to_opp)  # primary: net advantage; tie-break: our offense
                if cand > best_tuple:
                    best_tuple, best_idx = cand, i
            if best_idx is not None:
                onehot[best_idx] = 1.0
                return onehot
            # fallback if something weird
            onehot[valid_switch_idxs[0]] = 1.0
            return onehot

        # ----- decide: stay vs switch (very simple rule) -----
        # If our best move is neutral or better, just use it.
        if stay_best_a is not None and stay_best_eff >= 1.0:
            onehot[stay_best_a] = 1.0
            return onehot

        # Otherwise, consider switching if someone has super-effective coverage and isn't a punching bag.
        best_sw_idx, best_sw_score = None, -1e9
        for i in valid_switch_idxs:
            mon = team_list[i]
            new_types = getattr(mon, "types", []) or []
            my_to_opp  = best_stab_like(new_types, opp_types)     # 0..4 (we want >1)
            opp_to_new = best_stab_like(opp_types, new_types)     # 0..4 (we want small)
            # simple net advantage score
            sw_score = (my_to_opp) - (opp_to_new)
            if sw_score > best_sw_score:
                best_sw_score, best_sw_idx = sw_score, i

        # If a switch clearly improves (has SE coverage and not awful defense), switch; else stay.
        if best_sw_idx is not None:
            # require some minimum improvement over staying being resisted
            if (stay_best_a is None) or (best_sw_score > 0.5):  # 0.5 ~ "usually a worthwhile edge"
                onehot[best_sw_idx] = 1.0
                return onehot

        # default: stay and use our best move (even if resisted)
        if stay_best_a is not None:
            onehot[stay_best_a] = 1.0
            return onehot

        # final fallback: any switch or any move
        if valid_switch_idxs:
            onehot[valid_switch_idxs[0]] = 1.0
        elif valid_moves:
            onehot[valid_moves[0][0]] = 1.0
        return onehot

    # ---------- observation ----------
    def _observation_size(self) -> int:
        return 10

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        onehot = self._action_hint_onehot(battle).astype(np.float32)
        #print(onehot)
        return onehot

    # ---------- reward ----------
    # def calc_reward(self, battle: AbstractBattle) -> float:
    #     """
    #     Damage-first + hint-alignment:
    #       + 3.0*dmg + 2.0*dmg^2  (dominant; dmg ∈ [0..1] per mon)
    #       + 1.0 per KO
    #       + 1.0 if chosen action == hinted action (based on prior state)
    #       + 0.7 if we switched (action<=5) after a low-dmg turn (< low_damage_thresh)
    #     """
    #     try:
    #         prior = self._get_prior_battle(battle)
    #     except AttributeError:
    #         prior = None

    #     if prior is None:
    #         return 0.0

    #     reward = 0.0

    #     # Dense damage
    #     opp_prev = self._team_hp_sum(prior.opponent_team)
    #     opp_now  = self._team_hp_sum(battle.opponent_team)
    #     dmg_dealt = max(0.0, opp_prev - opp_now)
    #     reward += 3.0 * dmg_dealt + 2.0 * (dmg_dealt ** 2)
    #     if dmg_dealt >= 0.70:
    #         reward += 0.5

    #     # KO bonus
    #     def faint_count(team_dict):
    #         return sum(
    #             1 for mon in team_dict.values()
    #             if mon and (getattr(mon, "fainted", False) or getattr(mon, "current_hp", 0) <= 0)
    #         )
    #     opp_faints_prev = faint_count(prior.opponent_team)
    #     opp_faints_now  = faint_count(battle.opponent_team)
    #     if opp_faints_now > opp_faints_prev:
    #         reward += 1.0 * (opp_faints_now - opp_faints_prev)

    #     # Hint alignment (hint computed on prior state)
    #     hint_prior = self._action_hint_onehot(prior)
    #     hinted_idx = int(np.argmax(hint_prior)) if hint_prior.sum() > 0 else None
    #     if hinted_idx is not None and self._last_action is not None:
    #         if int(self._last_action) == hinted_idx:
    #             reward += 1.0

    #     # Switch after low damage
    #     if self._last_action is not None and int(self._last_action) <= 5:
    #         if dmg_dealt < self.low_damage_thresh:
    #             reward += 0.7

    #     # Tiny terminal positives (optional, very light)
    #     if battle.finished:
    #         my_now  = self._team_hp_sum(battle.team)
    #         hp_margin = max(0.0, my_now - opp_now)
    #         my_alive = sum(
    #             1 for m in battle.team.values()
    #             if m and not getattr(m, "fainted", False) and getattr(m, "current_hp", 0) > 0
    #         )
    #         reward += 0.10 * my_alive + 0.05 * hp_margin

    #     return max(0.0, round(float(reward), 4))
    
    # ---------- reward ----------
    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Pure hint imitation:
        +1.0 if the chosen action equals the hint computed on the PRIOR state
        +0.0 otherwise
        """
        try:
            prior = self._get_prior_battle(battle)
        except AttributeError:
            prior = None

        if prior is None:
            return 0.0  # no reward on the very first step

        hint_prior = self._action_hint_onehot(prior)
        hinted_idx = int(np.argmax(hint_prior)) if hint_prior.sum() > 0 else None

        if hinted_idx is not None and self._last_action is not None:
            return 1.0 if int(self._last_action) == hinted_idx else 0.0

        return 0.0


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