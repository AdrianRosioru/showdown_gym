import os
from typing import Any, Dict

import numpy as np
from poke_env import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.player import Player

from showdown_gym.base_environment import BaseShowdownEnv

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
        # Accepts poke_env PokemonType (has .name) or plain strings
        return (t.name if hasattr(t, "name") else str(t)).lower()

def offensive_multiplier(attacker_types, defender_types) -> float:
    """Product of effectiveness for all attacker types against all defender types."""
    mult = 1.0
    if not attacker_types or not defender_types:
        return mult  # neutral if something is unknown
    for atk in attacker_types:
        a = _type_name(atk)
        #print("Attack type: " + a)
        row = TYPE_CHART.get(a, {})
        for df in defender_types:
            d = _type_name(df)
            #print("Defender type: " + d)
            #print(row.get(d, 1.0))
            mult *= row.get(d, 1.0)
    return mult

def normalized_offensive_multiplier(attacker_types, defender_types) -> float:
    """Normalize to [0,1] by dividing by 4 (max is 4x)."""
    return offensive_multiplier(attacker_types, defender_types) / 4.0

class ShowdownEnvironment(BaseShowdownEnv):

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

        # Add any additional information you want to include in the info dictionary that is saved in logs
        # For example, you can add the win status

        if self.battle1 is not None:
            agent = self.possible_agents[0]
            info[agent]["win"] = self.battle1.won

        return info

    def _observation_size(self) -> int:
        # See embed_battle for the breakdown. If you tweak features, update this.
        return 116
    
    def stage_mult(self, stage: int) -> float:
        s = int(stage)
        return (2 + s) / 2.0 if s >= 0 else 2.0 / (2 - s)

    def approx_speed(self, mon):
        if not mon:
            return 1.0
        base_stats = getattr(mon, "base_stats", {}) or {}
        base_spe = float(base_stats.get("spe", 50.0))
        boosts = getattr(mon, "boosts", {}) or {}
        mult = self.stage_mult(int(boosts.get("spe", 0)))
        return base_spe * mult

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        # -----------------------
        # helpers (safe + generic)
        # -----------------------
        N_MOVES = 4
        MAX_TEAM = 6
        TYPES = [
            "normal","fire","water","electric","grass","ice","fighting","poison",
            "ground","flying","psychic","bug","rock","ghost","dragon","dark","steel","fairy"
        ]
        STAT_ORDER = ["atk","def","spa","spd","spe"]
        STATUS_KEYS = ["brn","psn","tox","par","slp","frz"]
        WEATHER_KEYS = ["sun", "rain", "sand", "snow", "none"]
        TERRAIN_KEYS = ["electric", "grassy", "psychic", "misty", "none"]

        def pad(lst, size, fill):
            return lst + [fill] * max(0, size - len(lst))

        def hp_from_slots(slots):
            out = []
            for mon in slots:
                out.append(float(getattr(mon, "current_hp_fraction", 1.0)) if mon else 1.0)
            return out

        def remaining_count(team_dict):
            # count mons with hp > 0 (or not fainted)
            cnt = 0
            for mon in team_dict.values():
                if mon is None:
                    continue
                if getattr(mon, "fainted", False):
                    continue
                if getattr(mon, "current_hp", 0) <= 0:
                    continue
                cnt += 1
            return cnt

        def one_hot_types(types):
            # types can be None/[], a single type, or dual
            present = set((_type_name(t) for t in (types or []) if t))
            return [1.0 if t in present else 0.0 for t in TYPES]

        def boosts_vec(mon):
            # map boost stage [-6..+6] -> [0,1] via (b+6)/12
            res = []
            boosts = getattr(mon, "boosts", {}) if mon else {}
            for k in STAT_ORDER:
                b = int(boosts.get(k, 0))
                res.append((b + 6) / 12.0)
            return res

        def status_onehot(mon):
            name = getattr(mon, "status", None)
            name = str(name).lower() if name else None
            vec = []
            for s in STATUS_KEYS:
                vec.append(1.0 if (name == s) else 0.0)
            return vec

        def side_hazards(side_conditions) -> Dict[str, float]:
            # Normalize spikes (0..3 -> 0..1), tspikes (0..2 -> 0..1), others bool {0,1}
            scl = {
                "stealthrock": 1.0,
                "spikes": 1/3.0,
                "toxicspikes": 1/2.0,
                "stickyweb": 1.0,
            }
            result = {k: 0.0 for k in scl}
            for key in list(scl.keys()):
                # poke-env uses enums; be robust to both names and raw strings
                val = 0
                for sc_key, sc_val in (side_conditions or {}).items():
                    name = str(getattr(sc_key, "name", sc_key)).lower()
                    if name == key:
                        # sc_val may be layer count or True
                        try:
                            val = int(sc_val)
                        except Exception:
                            val = 1 if sc_val else 0
                        break
                result[key] = float(val) * scl[key]
            return result

        def weather_onehot(battle):
            w = getattr(battle, "weather", None)
            wname = str(getattr(w, "name", w)).lower() if w else "none"
            vec = []
            for k in WEATHER_KEYS:
                vec.append(1.0 if (wname == k) else 0.0)
            return vec

        def terrain_onehot(battle):
            t = getattr(battle, "terrain", None)
            tname = str(getattr(t, "name", t)).lower() if t else "none"
            vec = []
            for k in TERRAIN_KEYS:
                vec.append(1.0 if (tname == k) else 0.0)
            return vec

        def normalized_bp(move):
            # normalize base power to [0,1] with 150 as a reasonable cap (randbats)
            if not move:
                return 0.0
            bp = float(getattr(move, "base_power", 0.0) or 0.0)
            return max(0.0, min(bp / 150.0, 1.0))

        def pp_frac(move):
            if not move:
                return 0.0
            max_pp = float(getattr(move, "max_pp", 0.0) or 0.0)
            cur_pp = float(getattr(move, "current_pp", 0.0) or 0.0)
            return 0.0 if max_pp <= 0 else max(0.0, min(cur_pp / max_pp, 1.0))

        # -----------------------
        # assemble slots & basics
        # -----------------------
        team_slots = list(battle.team.values())[:6]
        opp_slots  = list(battle.opponent_team.values())[:6]
        team_slots = pad(team_slots, MAX_TEAM, None)
        opp_slots  = pad(opp_slots,  MAX_TEAM, None)

        team_hp = hp_from_slots(team_slots)                  # 6
        opp_hp  = hp_from_slots(opp_slots)                   # 6
        team_rem = remaining_count(battle.team) / MAX_TEAM   # 1
        opp_rem  = remaining_count(battle.opponent_team) / MAX_TEAM  # 1

        my_active  = getattr(battle, "active_pokemon", None)
        opp_active = getattr(battle, "opponent_active_pokemon", None)

        my_types  = one_hot_types(getattr(my_active, "types", []) if my_active else [])
        opp_types = one_hot_types(getattr(opp_active, "types", []) if opp_active else [])

        my_boosts  = boosts_vec(my_active)
        opp_boosts = boosts_vec(opp_active)

        my_status  = status_onehot(my_active)
        opp_status = status_onehot(opp_active)

        my_speed  = self.approx_speed(my_active)
        opp_speed = self.approx_speed(opp_active)
        speed_edge = 1.0 if my_speed > opp_speed else 0.0
        # clamp ratio to [0, 3], then /3 to [0,1]
        speed_ratio = max(0.0, min(my_speed / max(1e-6, opp_speed), 3.0)) / 3.0

        # matchup scalars (current on-field)
        my_vs_opp = normalized_offensive_multiplier(
            getattr(my_active, "types", []) if my_active else [],
            getattr(opp_active, "types", []) if opp_active else []
        )
        opp_vs_my = normalized_offensive_multiplier(
            getattr(opp_active, "types", []) if opp_active else [],
            getattr(my_active, "types", []) if my_active else []
        )

        # per-slot multipliers vs opponent active (your whole team’s offensive pressure)
        def_types = getattr(opp_active, "types", []) if opp_active else []
        team_multipliers = []
        for p in team_slots:
            atk_types = getattr(p, "types", []) if p else []
            team_multipliers.append(normalized_offensive_multiplier(atk_types, def_types))

        # moves: effectiveness + base power + pp
        move_effectiveness, move_bps, move_pps = [], [], []
        avail_moves = getattr(battle, "available_moves", []) or []
        for move in avail_moves[:N_MOVES]:
            mtype = getattr(move, "type", None)
            eff = offensive_multiplier([mtype] if mtype else [], def_types) if def_types else 1.0
            move_effectiveness.append(float(eff) / 4.0)  # normalize similar to multiplier
            move_bps.append(normalized_bp(move))
            move_pps.append(pp_frac(move))
        move_effectiveness = pad(move_effectiveness, N_MOVES, 0.0)
        move_bps          = pad(move_bps,          N_MOVES, 0.0)
        move_pps          = pad(move_pps,          N_MOVES, 0.0)

        # field state: hazards, weather, terrain
        my_side_haz  = side_hazards(getattr(battle, "side_conditions", {}))
        opp_side_haz = side_hazards(getattr(battle, "opponent_side_conditions", {}))
        haz_my  = [my_side_haz[k]  for k in ["stealthrock","spikes","toxicspikes","stickyweb"]]
        haz_opp = [opp_side_haz[k] for k in ["stealthrock","spikes","toxicspikes","stickyweb"]]
        weather_vec = weather_onehot(battle)
        terrain_vec = terrain_onehot(battle)

        # switchability (forced switch, and bench size)
        forced_to_switch = 1.0 if getattr(battle, "force_switch", False) else 0.0
        my_bench = max(0, remaining_count(battle.team) - 1) / (MAX_TEAM - 1)  # normalized
        opp_bench = max(0, remaining_count(battle.opponent_team) - 1) / (MAX_TEAM - 1)

        # -----------------------
        # final vector (116 dims)
        # -----------------------
        final = (
            team_hp +                             # 6
            opp_hp +                              # 6 -> 12
            [team_rem, opp_rem] +                 # 2 -> 14
            [my_vs_opp, opp_vs_my] +              # 2 -> 16
            team_multipliers +                    # 6 -> 22
            move_effectiveness +                  # 4 -> 26
            move_bps +                            # 4 -> 30
            move_pps +                            # 4 -> 34
            my_types +                            # 18 -> 52
            opp_types +                           # 18 -> 70
            my_boosts +                           # 5  -> 75
            opp_boosts +                          # 5  -> 80
            my_status +                           # 6  -> 86
            opp_status +                          # 6  -> 92
            [speed_edge, speed_ratio] +           # 2  -> 94
            weather_vec +                         # 5  -> 99
            terrain_vec +                         # 5  -> 104
            haz_my + haz_opp +                    # 8  -> 112
            [forced_to_switch, my_bench, opp_bench]  # 3  -> 115
        )
        # add a small constant bias term for NN stability (optional)
        final.append(1.0)                         # 116

        return np.array(final, dtype=np.float32)

    def calc_reward(self, battle: AbstractBattle) -> float:
        prior = self._get_prior_battle(battle)

        # --------
        # HP terms
        # --------
        team_hp_now = [mon.current_hp_fraction for mon in battle.team.values()]
        opp_hp_now  = [mon.current_hp_fraction for mon in battle.opponent_team.values()]
        # pad to equal length for safe diffs
        L = max(len(team_hp_now), len(opp_hp_now))
        team_hp_now = (team_hp_now + [1.0]*(L - len(team_hp_now)))[:L]
        opp_hp_now  = (opp_hp_now  + [1.0]*(L - len(opp_hp_now)))[:L]

        team_hp_prev, opp_hp_prev = [1.0]*L, [1.0]*L
        if prior is not None:
            th = [mon.current_hp_fraction for mon in prior.team.values()]
            oh = [mon.current_hp_fraction for mon in prior.opponent_team.values()]
            team_hp_prev = (th + [1.0]*(L - len(th)))[:L]
            opp_hp_prev  = (oh + [1.0]*(L - len(oh)))[:L]

        diff_opp = np.array(opp_hp_prev) - np.array(opp_hp_now)   # damage dealt
        diff_team = np.array(team_hp_prev) - np.array(team_hp_now) # damage taken

        reward = 0.0
        reward += float(np.sum(diff_opp))            # + for damaging opponent
        reward -= float(np.sum(diff_team))           # - for taking damage

        # -------------
        # KO / fainting
        # -------------
        def faint_count(team_dict):
            cnt = 0
            for mon in team_dict.values():
                if mon is None:
                    continue
                if getattr(mon, "fainted", False) or getattr(mon, "current_hp", 0) <= 0:
                    cnt += 1
            return cnt

        if prior is not None:
            opp_faint_prev = faint_count(prior.opponent_team)
            opp_faint_now  = faint_count(battle.opponent_team)
            my_faint_prev  = faint_count(prior.team)
            my_faint_now   = faint_count(battle.team)

            reward += 1.0 * max(0, opp_faint_now - opp_faint_prev)   # reward KOs
            reward -= 1.0 * max(0, my_faint_now  - my_faint_prev)    # penalize own faints

        # --------------
        # Status changes
        # --------------
        def status_award(now_team, prev_team, positive_for_me=True):
            # +0.2 for each NEW negative status on the opponent; -0.2 for each NEW on me
            def to_statuses(tdict):
                st = []
                for mon in tdict.values():
                    s = getattr(mon, "status", None)
                    st.append(str(s).lower() if s else None)
                return st
            prev = to_statuses(prev_team) if prev_team is not None else []
            now  = to_statuses(now_team)
            prev = (prev + [None]*(len(now)-len(prev)))[:len(now)]
            delta = 0.0
            harmful = {"brn","psn","tox","par","slp","frz"}
            for p, n in zip(prev, now):
                if (p in (None, "none")) and (n in harmful):
                    delta += 0.2 if positive_for_me else -0.2
            return delta

        if prior is not None:
            reward += status_award(battle.opponent_team, prior.opponent_team, True)
            reward += status_award(battle.team,          prior.team,          False)

        # ----------------
        # Hazards changes
        # ----------------
        def hazard_scalar(side_conditions):
            # weight SR=0.3, Spikes layer=0.15 each, TSpikes layer=0.2 each, Web=0.25
            w = {"stealthrock":0.3, "spikes":0.15, "toxicspikes":0.2, "stickyweb":0.25}
            total = 0.0
            for sc_key, sc_val in (side_conditions or {}).items():
                name = str(getattr(sc_key, "name", sc_key)).lower()
                if name in w:
                    try:
                        layers = int(sc_val)
                    except Exception:
                        layers = 1 if sc_val else 0
                    total += w[name] * layers
            return total

        if prior is not None:
            opp_haz_prev = hazard_scalar(getattr(prior,  "opponent_side_conditions", {}))
            opp_haz_now  = hazard_scalar(getattr(battle, "opponent_side_conditions", {}))
            my_haz_prev  = hazard_scalar(getattr(prior,  "side_conditions", {}))
            my_haz_now   = hazard_scalar(getattr(battle, "side_conditions", {}))

            reward += 0.05 * (opp_haz_now - opp_haz_prev)  # more hazards on opp side is good
            reward += 0.05 * (my_haz_prev - my_haz_now)    # fewer hazards on my side is good

        # ---------------------------
        # Matchup & speed improvement
        # ---------------------------
        def current_matchup(b):
            my = getattr(b, "active_pokemon", None)
            op = getattr(b, "opponent_active_pokemon", None)
            mv = normalized_offensive_multiplier(getattr(my, "types", []) if my else [],
                                                 getattr(op, "types", []) if op else [])
            ov = normalized_offensive_multiplier(getattr(op, "types", []) if op else [],
                                                 getattr(my, "types", []) if my else [])
            return mv, ov

        def current_speed_edge(b):
            my = getattr(b, "active_pokemon", None)
            op = getattr(b, "opponent_active_pokemon", None)
            return 1.0 if self.approx_speed(my) > self.approx_speed(op) else 0.0

        mv_now, ov_now = current_matchup(battle)
        if prior is not None:
            mv_prev, ov_prev = current_matchup(prior)
            reward += 0.1 * (mv_now - mv_prev)          # improved my pressure
            reward += 0.1 * (ov_prev - ov_now)          # reduced their pressure

            se_now  = current_speed_edge(battle)
            se_prev = current_speed_edge(prior)
            reward += 0.05 * (se_now - se_prev)         # gained/lost speed edge

        # ----------------
        # Terminal outcome
        # ----------------
        if battle.finished:
            # strong terminal reward to align with true objective
            reward += 20.0 if battle.won else -20.0

        # ----------------
        # Small step cost
        # ----------------
        reward -= 0.01

        # clip to a reasonable band
        return float(np.clip(reward, -20.0, 20.0))


########################################
# DO NOT EDIT THE CODE BELOW THIS LINE #
########################################


class SingleShowdownWrapper(SingleAgentWrapper):
    """
    A wrapper class for the PokeEnvironment that simplifies the setup of single-agent
    reinforcement learning tasks in a Pokémon battle environment.

    This class initializes the environment with a specified battle format, opponent type,
    and evaluation mode. It also handles the creation of opponent players and account names
    for the environment.

    Do NOT edit this class!

    Attributes:
        battle_format (str): The format of the Pokémon battle (e.g., "gen9randombattle").
        opponent_type (str): The type of opponent player to use ("simple", "max", "random").
        evaluation (bool): Whether the environment is in evaluation mode.
    Raises:
        ValueError: If an unknown opponent type is provided.
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
