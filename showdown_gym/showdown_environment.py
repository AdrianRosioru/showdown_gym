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
    #print((t.name if hasattr(t, "name") else str(t)).lower())
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
    
    # def process_action(self, action: np.int64) -> np.int64:
    #     """
    #     Ignore 'action' and execute the action indicated by the current onehot hint.
    #     Maps: 0..5 = switch slots, 6..9 = moves 0..3.
    #     Falls back to first valid move, then first valid switch, else default (-2).
    #     """
    #     # Try to get the live battle for the learning agent
    #     battle = getattr(self, "battle1", None)

    #     # If we can't see a battle yet, just pass through
    #     if battle is None:
    #         try:
    #             self._last_action = int(action)
    #         except Exception:
    #             self._last_action = None
    #         return action

    #     # Compute the onehot hint and pick its argmax
    #     onehot = self._action_hint_onehot(battle)
    #     hinted_idx = int(np.argmax(onehot)) if isinstance(onehot, np.ndarray) and onehot.size == 10 else None

    #     # Build current legal action sets
    #     valid_moves = []
    #     avail_moves = (getattr(battle, "available_moves", []) or [])[:4]
    #     for mi, mv in enumerate(avail_moves):
    #         if not bool(getattr(mv, "disabled", False)):
    #             valid_moves.append(6 + mi)

    #     valid_switches = []
    #     my_act = getattr(battle, "active_pokemon", None)
    #     team = list(getattr(battle, "team", {}).values())[:6]
    #     for i, mon in enumerate(team):
    #         if mon is None or mon is my_act:
    #             continue
    #         if not getattr(mon, "fainted", False) and (getattr(mon, "current_hp", 0) > 0):
    #             valid_switches.append(i)

    #     valid_actions = set(valid_moves) | set(valid_switches)

    #     # Prefer the hinted action if it's legal
    #     if hinted_idx is not None and hinted_idx in valid_actions:
    #         self._last_action = hinted_idx
    #         return np.int64(hinted_idx)

    #     # Otherwise: first valid move, then first valid switch, else default
    #     if valid_moves:
    #         a = valid_moves[0]
    #         self._last_action = a
    #         return np.int64(a)

    #     if valid_switches:
    #         a = valid_switches[0]
    #         self._last_action = a
    #         return np.int64(a)

    #     # No legal actions detected (should be rare) -> default
    #     self._last_action = None
    #     return np.int64(-2)

    # ---------- helpers (reuses your type/matchup utils) ----------
    @staticmethod
    def _team_hp_sum(team_dict):
        return sum(float(getattr(mon, "current_hp_fraction", 0.0) or 0.0)
                   for mon in team_dict.values())

    def _battle_key(self, battle: AbstractBattle) -> str:
        return getattr(battle, "battle_tag", str(id(battle)))

    def _action_hint_onehot(self, battle: AbstractBattle) -> np.ndarray:
        """
        Minimal, robust heuristic (immunity-aware + risk-aware + boost/status-aware):
        - Scores moves: bp * eff * STAB * acc * risk_mult * my_offense_mult (per-phys/special).
        - Finisher: same as before but using adjusted opp threat.
        - Switch: resisted/immune/low-damage or danger, but discourage switching out if we're
        highly boosted in offense or speed (keep the sweep).
        """
        onehot = np.zeros(10, dtype=np.float32)

        # ----- helpers (unchanged + new) -----
        def eff_vs(atk_type, defender_types) -> float:
            if not atk_type or not defender_types: return 1.0
            row = TYPE_CHART.get(_type_name(atk_type), {})
            mult = 1.0
            for df in defender_types:
                mult *= row.get(_type_name(df), 1.0)
            return float(mult)

        def blocks_type(defender, atk_type) -> bool:
            if not defender or not atk_type: return False
            t = _type_name(atk_type)
            item = (getattr(defender, "item", None) or "").strip().lower()
            if item == "airballoon" and t == "ground": return True
            ab = (getattr(defender, "ability", None) or "").strip().lower()
            if ab == "levitate" and t == "ground": return True
            if ab in ("flashfire",) and t == "fire": return True
            if ab in ("voltabsorb","lightningrod","motordrive") and t == "electric": return True
            if ab in ("waterabsorb","stormdrain","dryskin") and t == "water": return True
            if ab in ("sapsipper",) and t == "grass": return True
            if ab == "wonderguard":
                eff = eff_vs(atk_type, getattr(defender, "types", []) or [])
                return eff <= 1.0
            return False

        def best_stab_like(attacker_types, defender_types) -> float:
            if not attacker_types or not defender_types: return 1.0
            best = 0.0
            for atk in attacker_types:
                best = max(best, eff_vs(atk, defender_types))
            return best

        def best_stab_like_passive(attacker_types, defender) -> float:
            if not attacker_types or not defender: return 1.0
            dtypes = getattr(defender, "types", []) or []
            best = 0.0
            for atk in attacker_types:
                eff = 0.0 if blocks_type(defender, atk) else eff_vs(atk, dtypes)
                best = max(best, eff)
            return best

        def is_alive(mon) -> bool:
            return bool(mon and not getattr(mon, "fainted", False) and (getattr(mon, "current_hp", 0) > 0))

        def safe_accuracy(mv) -> float:
            acc = None
            entry = getattr(mv, "entry", None)
            if isinstance(entry, dict): acc = entry.get("accuracy", None)
            if acc in (None, True):
                try: acc = getattr(mv, "accuracy", None)
                except Exception: acc = None
            if acc in (None, True): return 1.0
            try:
                acc = float(acc)
                return acc/100.0 if acc > 1.0 else max(0.0, min(1.0, acc))
            except Exception:
                return 1.0

        def safe_priority(mv) -> int:
            pr = 0
            try: pr = int(getattr(mv, "priority", 0) or 0)
            except Exception: pr = 0
            if pr == 0:
                entry = getattr(mv, "entry", None)
                if isinstance(entry, dict):
                    try: pr = int(entry.get("priority", 0) or 0)
                    except Exception: pr = 0
            return pr

        def hp_ratio(poke) -> float:
            try:
                cur = float(getattr(poke, "current_hp", None) or 0.0)
                mx  = getattr(poke, "max_hp", None)
                if mx is None:
                    stats = getattr(poke, "stats", {}) or {}
                    mx = stats.get("hp", None)
                mx = float(mx) if mx not in (None, 0) else None
                if not mx: return 1.0
                return max(0.0, min(1.0, cur / mx))
            except Exception:
                return 1.0

        # --- NEW: status normalization (handles enums) ---
        def normalize_status(x) -> str:
            """Return 'brn','par','psn','tox','slp','frz' or ''."""
            if x is None or x is False:
                return ""
            if isinstance(x, str):
                return x.lower()
            name = getattr(x, "name", None)
            if name is not None:
                return str(name).lower()
            value = getattr(x, "value", None)
            if value is not None:
                return str(value).lower()
            return str(x).lower()

        # --- NEW: stage → multiplier ---
        def stage_mult(stage: int) -> float:
            # Pokémon stage formula: positive: (2+stg)/2, negative: 2/(2-stg)
            try: s = int(stage)
            except Exception: s = 0
            s = max(-6, min(6, s))
            if s >= 0: return (2.0 + s) / 2.0
            return 2.0 / (2.0 - s)

        # --- NEW: get my/opp boost + status multipliers (None-safe) ---
        def stat_context(attacker, defender):
            """Stage multipliers for attacker (Atk/SpA/Spe) and defender (Def/SpD), status-aware."""
            if attacker is None or defender is None:
                return 1.0, 1.0, 1.0, 1.0, 1.0

            boosts_a = getattr(attacker, "boosts", {}) or {}
            boosts_d = getattr(defender, "boosts", {}) or {}
            atk_mul = stage_mult(boosts_a.get("atk", 0))
            spa_mul = stage_mult(boosts_a.get("spa", 0))
            def_mul = stage_mult(boosts_d.get("def", 0))
            spd_mul = stage_mult(boosts_d.get("spd", 0))
            spe_mul = stage_mult(boosts_a.get("spe", 0))

            # Conservative status effects
            status_a = normalize_status(getattr(attacker, "status", None))
            if status_a == "brn":   # burn halves physical damage
                atk_mul *= 0.5
            elif status_a == "par": # gen8/9: speed * 0.5
                spe_mul *= 0.5

            return atk_mul, spa_mul, def_mul, spd_mul, spe_mul

        # ----- state -----
        my_act  = getattr(battle, "active_pokemon", None)
        opp_act = getattr(battle, "opponent_active_pokemon", None)
        my_types  = getattr(my_act, "types", []) if my_act else []
        opp_types = getattr(opp_act, "types", []) if opp_act else []

        # opponent threat vs current mon (type-only first)
        opp_to_me = best_stab_like_passive(opp_types, my_act)

        # --- threat adjusted by boosts/status ---
        o_atk_mul, o_spa_mul, my_def_mul, my_spd_mul, _ = stat_context(opp_act, my_act)
        opp_offense_mul = max(o_atk_mul / max(1e-9, my_def_mul), o_spa_mul / max(1e-9, my_spd_mul))
        opp_threat = opp_to_me * max(1.0, opp_offense_mul)  # never scale down; only up

        # action sets
        team_list = list(battle.team.values())[:6]
        valid_switch_idxs = [i for i, mon in enumerate(team_list) if (mon is not None and mon is not my_act and is_alive(mon))]
        avail_moves = (getattr(battle, "available_moves", []) or [])[:4]
        valid_moves = [(6 + mi, mv) for mi, mv in enumerate(avail_moves) if not bool(getattr(mv, "disabled", False))]

        # thresholds / knobs
        low_score_thresh = float(getattr(self, "low_move_score_thresh", 60.0))
        resisted_eff_thresh = float(getattr(self, "resisted_eff_thresh", 1.0))
        finisher_hp_thresh = float(getattr(self, "finisher_hp_thresh", 0.15))
        min_move_acc = float(getattr(self, "min_move_acc", 0.80))
        risk_mult_lowacc = float(getattr(self, "risk_mult_lowacc", 0.5))
        risk_mult_chargetype = float(getattr(self, "risk_mult_chargetype", 0.6))

        # my offensive/speed multipliers for move scoring & “don’t swap” bias
        my_atk_mul, my_spa_mul, _d_, _s_, my_spe_mul = stat_context(my_act, opp_act)
        my_big_offense = max(my_atk_mul, my_spa_mul)
        my_sweepy_speed = my_spe_mul

        # ----- score moves (risk-aware + boost-aware) -----
        stay_best_a, stay_best_score, stay_best_eff = None, -1.0, 1.0
        finisher_choice, finisher_prio, finisher_acc = None, 0, 1.0
        finisher_key = (-1.0, -999, -1.0, -1.0, -1.0)  # (acc, prio, eff, stab, bp)

        any_nonimmune = False
        for a, mv in valid_moves:
            entry = getattr(mv, "entry", None)
            safe_name = str(getattr(mv, "id", getattr(mv, "name", ""))).lower()

            # failure-prone patterns
            is_ohko = safe_name in ("sheercold", "fissure", "horndrill", "guillotine")
            if (isinstance(entry, dict) and entry.get("ohko", False)): is_ohko = True
            two_turn = isinstance(entry, dict) and bool(
                entry.get("twoTurnMove") or entry.get("two_turn") or entry.get("chargingTurn")
                or (isinstance(entry.get("flags"), dict) and entry["flags"].get("charge", False))
            )
            # safe recharge detection
            recharge = False
            if isinstance(entry, dict):
                if bool(entry.get("recharge")): recharge = True
                else:
                    self_sec = entry.get("self")
                    if isinstance(self_sec, dict) and self_sec.get("volatileStatus") == "mustrecharge":
                        recharge = True

            # base components
            bp   = float(getattr(mv, "base_power", 0.0) or 0.0)
            mtyp = getattr(mv, "type", None)
            eff  = eff_vs(mtyp, opp_types) if opp_types else 1.0
            if opp_act and blocks_type(opp_act, mtyp): eff = 0.0
            if eff == 0.0: continue
            any_nonimmune = True

            stab = 1.5 if (mtyp and any(_type_name(mtyp) == _type_name(t) for t in (my_types or []))) else 1.0
            acc  = safe_accuracy(mv)
            prio = safe_priority(mv)

            if is_ohko:  # skip OHKO cheese
                continue

            # detect category and apply OUR boost/status to scoring
            cat = str(getattr(getattr(mv, "category", None), "name", getattr(mv, "category", ""))).lower()
            if cat == "physical":
                my_off_mult = my_atk_mul
            elif cat == "special":
                my_off_mult = my_spa_mul
            else:
                my_off_mult = 1.0  # status moves

            # risk shaping
            risk_mult = 1.0
            if acc < min_move_acc:
                risk_mult *= risk_mult_lowacc
            if two_turn or recharge:
                risk_mult *= (risk_mult_chargetype * (0.7 if opp_threat >= 2.0 else 1.0))

            score = bp * eff * stab * acc * risk_mult * my_off_mult

            if score > stay_best_score:
                stay_best_score = score
                stay_best_a = a
                stay_best_eff = eff

            if bp > 0.0:
                key = (acc, prio, eff, stab, bp)
                if key > finisher_key:
                    finisher_key = key
                    finisher_choice = a
                    finisher_prio = prio
                    finisher_acc  = acc

        immune_or_no_moves = (stay_best_a is None) or (not any_nonimmune)
        resisted = (stay_best_a is not None) and (stay_best_eff < resisted_eff_thresh)
        low_damage = (stay_best_a is not None) and (stay_best_score < low_score_thresh)

        # ----- FINISHER (safer; uses adjusted threat) -----
        opp_is_low = bool(opp_act) and (hp_ratio(opp_act) <= finisher_hp_thresh)
        if opp_is_low and finisher_choice is not None:
            if ((opp_threat < 2.0) or (finisher_prio > 0)) and (finisher_acc >= min_move_acc or finisher_prio > 0):
                onehot[finisher_choice] = 1.0
                return onehot

        # ----- no valid damaging move -> pick best switch now -----
        if immune_or_no_moves and valid_switch_idxs:
            best_idx, best_tuple = None, (-1e9, -1e9, -1e9)
            for i in valid_switch_idxs:
                mon = team_list[i]
                my_to_opp  = best_stab_like_passive(getattr(mon, "types", []) or [], opp_act)
                # Adjust incoming by boosts/status for new mon too
                o_atk_mul2, o_spa_mul2, def_mul2, spd_mul2, _ = stat_context(opp_act, mon)
                opp_to_new_base = best_stab_like_passive(opp_types, mon)
                opp_to_new = opp_to_new_base * max(1.0, max(o_atk_mul2 / max(1e-9, def_mul2),
                                                            o_spa_mul2 / max(1e-9, spd_mul2)))
                hpw = hp_ratio(mon)
                cand = (my_to_opp - opp_to_new, my_to_opp, hpw)
                if cand > best_tuple:
                    best_tuple, best_idx = cand, i
            if best_idx is not None:
                onehot[best_idx] = 1.0
                return onehot

        # ======= STAY vs SWAP (uses adjusted threat) =======

        # If neutral-or-better and not in big danger, stay (unless low damage triggers pivot below)
        if (stay_best_a is not None) and (stay_best_eff >= 1.0) and not (opp_threat >= 2.0):
            if not low_damage:
                onehot[stay_best_a] = 1.0
                return onehot

        # Build best switch candidate (usable for all triggers)
        best_sw_idx, best_sw_score = None, -1e9
        for i in valid_switch_idxs:
            mon = team_list[i]
            my_to_opp  = best_stab_like_passive(getattr(mon, "types", []) or [], opp_act)
            # adjusted incoming for new mon
            o_atk_mul2, o_spa_mul2, def_mul2, spd_mul2, _ = stat_context(opp_act, mon)
            opp_to_new_base = best_stab_like_passive(opp_types, mon)
            opp_to_new = opp_to_new_base * max(1.0, max(o_atk_mul2 / max(1e-9, def_mul2),
                                                        o_spa_mul2 / max(1e-9, spd_mul2)))

            # allow defensive pivots that shave a lot of incoming (e.g., 4x -> 2x)
            danger_reduction = opp_threat - opp_to_new
            good_offense = (my_to_opp > 1.0) or (my_to_opp >= 1.0 and opp_to_new <= 1.0) or (danger_reduction >= 1.0)
            if not good_offense:
                continue

            sw_score = (my_to_opp) - (opp_to_new)
            sw_score += 0.05 * hp_ratio(mon)  # tiny HP tiebreak
            if sw_score > best_sw_score:
                best_sw_score = sw_score
                best_sw_idx = i

        # Decide if we should actually swap
        should_consider_swap = immune_or_no_moves or resisted or low_damage or (opp_threat >= 2.0)

        if best_sw_idx is not None and should_consider_swap:
            # base threshold as before, but…
            if stay_best_a is None or immune_or_no_moves:
                need = 0.0
            elif resisted:
                need = 0.25 if stay_best_eff < 0.5 else 0.5
            elif low_damage:
                need = 0.25
            else:
                # danger pivot from neutral (use adjusted threat):
                need = max(0.0, 0.5 - 0.25 * (opp_threat - 2.0))

            # if we're highly boosted in offense or speed, require more to switch
            if my_big_offense >= 2.0:   # e.g., +2 Atk or +2 SpA
                need += 0.5
            if my_sweepy_speed >= 2.0:  # e.g., +2 Spe
                need += 0.5

            if best_sw_score >= need:
                onehot[best_sw_idx] = 1.0
                return onehot

        # OPTIONAL: auto-pivot when quadruple-weak (ultra-conservative, using adjusted threat)
        if opp_threat >= 4.0 and valid_switch_idxs:
            best_idx, best_tuple = None, (99.0, -1.0)  # (incoming, our_offense)
            for i in valid_switch_idxs:
                mon = team_list[i]
                my_to_opp  = best_stab_like_passive(getattr(mon, "types", []) or [], opp_act)
                o_atk_mul2, o_spa_mul2, def_mul2, spd_mul2, _ = stat_context(opp_act, mon)
                opp_to_new_base = best_stab_like_passive(opp_types, mon)
                opp_to_new = opp_to_new_base * max(1.0, max(o_atk_mul2 / max(1e-9, def_mul2),
                                                            o_spa_mul2 / max(1e-9, spd_mul2)))
                cand = (opp_to_new, my_to_opp)
                if cand < best_tuple:
                    best_tuple, best_idx = cand, i
            if best_idx is not None:
                onehot[best_idx] = 1.0
                return onehot

        # default: stay and use our best move
        if stay_best_a is not None:
            onehot[stay_best_a] = 1.0
            return onehot

        # last resort
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

    # def calc_reward(self, battle: AbstractBattle) -> float:
    #     """
    #     Imitation with value-aware shaping & streak bonus.

    #     Base:
    #     +1.0 if chosen action == hint(prior_state)
    #     -0.5 otherwise

    #     Shaping:
    #     + streak bonus for consecutive matches (0.1, 0.2, 0.3, ...), capped if configured.
    #     - regret penalty when mismatched, scaled by how much worse the chosen action's
    #         heuristic value is vs the hinted action's value on the PRIOR state.

    #     Notes:
    #     - Uses the same scoring you trust (bp * eff * STAB * acc for moves;
    #         my->opp - opp->my for switches).
    #     - Uses self._norm_scale to normalize regret (10.0 by default).
    #     - Optionally slaps illegal actions further (config flag below).
    #     """
    #     # ---- get prior state ----
    #     try:
    #         prior = self._get_prior_battle(battle)
    #     except AttributeError:
    #         prior = None
    #     if prior is None:
    #         return 0.0

    #     # ---- helper scorers on the PRIOR state (no deps outside your file) ----
    #     def eff_vs(atk_type, defender_types) -> float:
    #         if not atk_type or not defender_types:
    #             return 1.0
    #         row = TYPE_CHART.get(_type_name(atk_type), {})
    #         mult = 1.0
    #         for df in defender_types:
    #             mult *= row.get(_type_name(df), 1.0)
    #         return float(mult)

    #     def best_stab_like(attacker_types, defender_types) -> float:
    #         if not attacker_types or not defender_types:
    #             return 1.0
    #         best = 0.0
    #         for atk in attacker_types:
    #             best = max(best, eff_vs(atk, defender_types))
    #         return best

    #     def is_alive(mon) -> bool:
    #         return bool(mon and not getattr(mon, "fainted", False) and (getattr(mon, "current_hp", 0) > 0))

    #     def safe_accuracy(mv) -> float:
    #         acc = None
    #         entry = getattr(mv, "entry", None)
    #         if isinstance(entry, dict):
    #             acc = entry.get("accuracy", None)
    #         if acc in (None, True):
    #             try:
    #                 acc = getattr(mv, "accuracy", None)
    #             except Exception:
    #                 acc = None
    #         if acc in (None, True):
    #             return 1.0
    #         try:
    #             acc = float(acc)
    #             return acc / 100.0 if acc > 1.0 else max(0.0, min(1.0, acc))
    #         except Exception:
    #             return 1.0

    #     def legal_actions_on(b: AbstractBattle) -> set[int]:
    #         legal = set()
    #         # moves 6..9
    #         avail_moves = (getattr(b, "available_moves", []) or [])[:4]
    #         for mi, mv in enumerate(avail_moves):
    #             if not bool(getattr(mv, "disabled", False)):
    #                 legal.add(6 + mi)
    #         # switches 0..5
    #         my_act = getattr(b, "active_pokemon", None)
    #         team = list(getattr(b, "team", {}).values())[:6]
    #         for i, mon in enumerate(team):
    #             if mon is None or mon is my_act:
    #                 continue
    #             if is_alive(mon):
    #                 legal.add(i)
    #         return legal

    #     def move_value_on(b: AbstractBattle, idx: int) -> float:
    #         """Heuristic value for move action idx (6..9) on state b."""
    #         if idx < 6 or idx > 9:
    #             return -1e9
    #         mi = idx - 6
    #         avail_moves = (getattr(b, "available_moves", []) or [])[:4]
    #         if mi >= len(avail_moves):
    #             return -1e9
    #         mv = avail_moves[mi]
    #         if bool(getattr(mv, "disabled", False)):
    #             return -1e9
    #         bp   = float(getattr(mv, "base_power", 0.0) or 0.0)
    #         mtyp = getattr(mv, "type", None)
    #         opp  = getattr(b, "opponent_active_pokemon", None)
    #         opp_types = getattr(opp, "types", []) if opp else []
    #         myp  = getattr(b, "active_pokemon", None)
    #         my_types = getattr(myp, "types", []) if myp else []
    #         eff  = eff_vs(mtyp, opp_types) if opp_types else 1.0
    #         if eff == 0.0:
    #             return -1e9
    #         stab = 1.5 if (mtyp and any(_type_name(mtyp) == _type_name(t) for t in (my_types or []))) else 1.0
    #         acc  = safe_accuracy(mv)
    #         return bp * eff * stab * acc

    #     def switch_value_on(b: AbstractBattle, idx: int) -> float:
    #         """Heuristic value for switch action idx (0..5) on state b."""
    #         if idx < 0 or idx > 5:
    #             return -1e9
    #         team_list = list(getattr(b, "team", {}).values())[:6]
    #         if idx >= len(team_list):
    #             return -1e9
    #         mon = team_list[idx]
    #         if not is_alive(mon):
    #             return -1e9
    #         opp  = getattr(b, "opponent_active_pokemon", None)
    #         new_types = getattr(mon, "types", []) or []
    #         opp_types = getattr(opp, "types", []) if opp else []
    #         my_to_opp  = best_stab_like(new_types, opp_types)   # prefer higher
    #         opp_to_new = best_stab_like(opp_types, new_types)   # prefer lower
    #         return (my_to_opp - opp_to_new)

    #     def action_value_on(b: AbstractBattle, idx: Optional[int]) -> float:
    #         if idx is None:
    #             return -1e9
    #         return move_value_on(b, idx) if idx >= 6 else switch_value_on(b, idx)

    #     # ---- compute hint & compare ----
    #     hint_prior = self._action_hint_onehot(prior)
    #     hinted_idx = int(np.argmax(hint_prior)) if isinstance(hint_prior, np.ndarray) and hint_prior.size == 10 and hint_prior.sum() > 0 else None
    #     chosen_idx = int(self._last_action) if self._last_action is not None else None

    #     if hinted_idx is None or chosen_idx is None:
    #         return 0.0

    #     # (Optional) extra punishment for illegal actions
    #     legal = legal_actions_on(prior)
    #     illegal_penalty = -1.0 if chosen_idx not in legal else 0.0

    #     # value-aware regret shaping
    #     v_hint   = action_value_on(prior, hinted_idx)
    #     v_chosen = action_value_on(prior, chosen_idx)
    #     regret_raw = max(0.0, v_hint - v_chosen)  # only penalize if worse than hint
    #     regret = min(1.0, regret_raw / float(getattr(self, "_norm_scale", 10.0)))

    #     # streak handling
    #     key = self._battle_key(battle)
    #     if key not in self._streak_bonus:
    #         self._streak_bonus[key] = float(getattr(self, "_streak_bonus_init", 0.1) or 0.0)
    #     streak_step = float(getattr(self, "_streak_bonus_step", 0.1) or 0.0)
    #     streak_cap  = getattr(self, "_streak_bonus_cap", None)

    #     # base alignment reward
    #     if chosen_idx == hinted_idx:
    #         # match
    #         r = 1.0
    #         # add & grow streak
    #         r += self._streak_bonus[key]
    #         self._streak_bonus[key] += streak_step
    #         if streak_cap is not None:
    #             self._streak_bonus[key] = min(self._streak_bonus[key], float(streak_cap))
    #         # tiny positive nudge if the hint itself was very strong (optional)
    #         if v_hint > 0:
    #             r += 0.2 * min(1.0, v_hint / float(getattr(self, "_norm_scale", 10.0)))
    #     else:
    #         # mismatch
    #         r = -0.5
    #         r -= 0.8 * regret   # more negative if we ignored a much-better hint
    #         r += illegal_penalty
    #         # reset streak
    #         self._streak_bonus[key] = float(getattr(self, "_streak_bonus_init", 0.1) or 0.0)

    #     # optional: tiny terminal summary nudge so episodes with high imitation ratio pay out more
    #     if battle.finished:
    #         # imitation ratio proxy from history if you track it; here we add nothing to keep it minimal.
    #         pass

    #     return float(round(r, 4))
    
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