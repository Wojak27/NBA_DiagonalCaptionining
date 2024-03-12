import json
from nba_api.stats.endpoints import commonplayerinfo, playercareerstats
import pandas as pd
import pickle

# Assume we have a JSON file 'player_list.json' with the player IDs list
# with open('/home/karolwojtulewicz/code/NSVA/all_players.json', 'r') as file:
#     data = json.load(file)
#     player_ids_with_prefix = data

with open('/home/karolwojtulewicz/code/NSVA/all_unique_players.pickle', 'rb') as file:
    data = pickle.load(file)
    player_ids_with_prefix = [key for key in data.keys() if key.startswith('PLAYER')]

# Step 1: Clean player IDs
player_ids = set(player_id.replace('PLAYER', '').replace(':', '').replace("side", "").replace(",", "") for player_id in player_ids_with_prefix)

# Step 2: Initialize dictionary to store player info
player_info_dict = {}

# Step 3: Iterate over player IDs and fetch data
for player_id in player_ids:
    try:
        career = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        stats = playercareerstats.PlayerCareerStats(player_id=player_id)
        player_info = career.get_normalized_dict()
        common_info = player_info['CommonPlayerInfo'][0]  # assuming the first entry is what we want
        stats_info = stats.get_normalized_dict()
        stats_info = stats_info['SeasonTotalsRegularSeason'][[True if "2018-19" in s else False for s in stats_info['SeasonTotalsRegularSeason']][0] ]  # assuming the first entry is what we want

        # Extract required information
        info_dict = {
            'first_name': common_info['FIRST_NAME'],
            'last_name': common_info['LAST_NAME'],
            'height': common_info['HEIGHT'],
            'jersey': common_info['JERSEY'],
            'age': stats_info['PLAYER_AGE'],
            'FG_PCT': stats_info['FG_PCT'],
            'AST': stats_info['AST']/(stats_info['GP'] or 1),
            'FT_PCT': stats_info['FT_PCT'],
            'REB': stats_info['REB']/(stats_info['GP'] or 1),
            'FG3_PCT': stats_info['FG3_PCT'],
            'STL': stats_info['STL']/(stats_info['GP'] or 1),
            'PTS': stats_info['PTS']/(stats_info['GP'] or 1),
            'BLK': stats_info['BLK']/(stats_info['GP'] or 1),
            
        }

        # Step 4: Add to the main dictionary
        player_info_dict[player_id] = info_dict

    except Exception as e:
        print(f"Failed to fetch data for Player ID {player_id}: {e}")

# Save the results to a JSON file
with open('player_info_dict.json', 'w') as outfile:
    json.dump(player_info_dict, outfile, indent=4)

print("Finished saving player information to player_info_dict.json")