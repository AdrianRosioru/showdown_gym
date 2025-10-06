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

    def calc_reward(self, battle: AbstractBattle) -> float:
        prior_battle = self._get_prior_battle(battle)
        reward = 0.0

        # --- Current health ---
        health_team = [mon.current_hp_fraction for mon in battle.team.values()]
        health_opponent = [mon.current_hp_fraction for mon in battle.opponent_team.values()]

        if len(health_opponent) < len(health_team):
            health_opponent.extend([1.0] * (len(health_team) - len(health_opponent)))

        # --- Prior health ---
        prior_health_team, prior_health_opponent = [], []
        if prior_battle is not None:
            prior_health_team = [
                mon.current_hp_fraction for mon in prior_battle.team.values()
            ]
            prior_health_opponent = [
                mon.current_hp_fraction for mon in prior_battle.opponent_team.values()
            ]

        # Pad prior lists
        if len(prior_health_team) < len(health_team):
            prior_health_team.extend([1.0] * (len(health_team) - len(prior_health_team)))
        if len(prior_health_opponent) < len(health_team):
            prior_health_opponent.extend([1.0] * (len(health_team) - len(prior_health_opponent)))

        # --- Diffs ---
        diff_health_opponent = np.array(prior_health_opponent) - np.array(health_opponent)
        diff_health_team = np.array(prior_health_team) - np.array(health_team)

        # --- Rewards ---
        reward += np.sum(diff_health_opponent)   # reward for damaging opponent
        reward -= np.sum(diff_health_team)       # penalty for losing own HP

        # --- Clip to [-6, +6] ---
        reward = float(np.clip(reward, -6.0, 6.0))

        return reward


    def _observation_size(self) -> int:
        """
        Returns the size of the observation size to create the observation space for all possible agents in the environment.

        You need to set obvervation size to the number of features you want to include in the observation.
        Annoyingly, you need to set this manually based on the features you want to include in the observation from emded_battle.

        Returns:
            int: The size of the observation space.
        """

        # Simply change this number to the number of features you want to include in the observation from embed_battle.
        # If you find a way to automate this, please let me know!
        return 22

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        N_MOVES = 4

        def pad(lst, size, fill):
            return lst + [fill] * max(0, size - len(lst))

        # --- slots from the already-ordered dicts (preserve order) ---
        team_slots = list(battle.team.values())[:6]
        opp_slots  = list(battle.opponent_team.values())[:6]
        team_slots = pad(team_slots, 6, None)
        opp_slots  = pad(opp_slots, 6, None)

        # --- HP blocks in the SAME order as the dicts ---
        def hp_from_slots(slots):
            hp = []
            for mon in slots:
                if mon is None:
                    hp.append(1.0)  # keep your convention; switch to 0.0 if you prefer
                else:
                    hp.append(float(getattr(mon, "current_hp_fraction", 0.0)))
            return hp

        team_hp = hp_from_slots(team_slots)   # length 6, aligned to dict order
        opp_hp  = hp_from_slots(opp_slots)    # length 6

        # --- opponent defensive types (for multipliers/effectiveness) ---
        opp_active = getattr(battle, "opponent_active_pokemon", None)
        def_types  = (opp_active.types or []) if opp_active else []

        # --- per-slot multipliers (length 6) in SAME order as team_hp ---
        def slot_multiplier(pkmn, def_types):
            if not pkmn or not def_types:
                return 0.0
            atk_types = pkmn.types or []
            return normalized_offensive_multiplier(atk_types, def_types)

        team_multipliers = [slot_multiplier(p, def_types) for p in team_slots]

        # --- move effectiveness (not damage) for current active (length 4) ---
        move_effectiveness = []
        active = getattr(battle, "active_pokemon", None)
        if active and opp_active and def_types:
            for move in getattr(battle, "available_moves", []):
                mtype = getattr(move, "type", None)
                eff = offensive_multiplier([mtype] if mtype else [], def_types)
                move_effectiveness.append(float(eff))
        move_effectiveness = pad(move_effectiveness, N_MOVES, 0.0)

        # --- final vector: team_hp (6) + opp_hp (6) + team_multipliers (6) + move_eff (4) ---
        final_vector = np.array(
            team_hp + opp_hp + team_multipliers + move_effectiveness,
            dtype=np.float32,
        )

        # pretty-print: 6 columns
        # for i in range(0, final_vector.size, 6):
        #     print(" ".join(f"{x:.6g}" for x in final_vector[i:i+6]))

        return final_vector

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
