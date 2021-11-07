from numpy.lib.function_base import angle
import pandas as pd
import json
from math import atan2

data = json.load(open('shots.json', 'r'))

num_shots = len(data)
CENTRE_OF_GOAL = (120, 40)
LEFT_OF_GOAL = (120, 36)
RIGHT_OF_GOAL = (120, 44)

def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

def player_between_goal(shot_location, player_location):
    pt = player_location
    v1, v2, v3 = shot_location, RIGHT_OF_GOAL, LEFT_OF_GOAL
    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    if has_neg and has_pos:
        return 0
    else:
        return 1

play_pattern = []
duration = []
distance_to_goal = []
angle_to_goal = []
players_between_goal = []
players_within_1 = []
players_within_5 = []
players_within_10 = []
opponents_within_5 = []
teammates_within_5 = []
shot_taker_type = []
shot_technique = []
shot_body_part = []
goalkeeper_in_the_way = []
one_on_one = []

goal = []


for shot in data:
    if 'freeze_frame' not in shot['shot'].keys():
        continue

    position_name = shot['position']['name'].lower()
    regular_position_type = None
    if 'forward' in position_name or 'striker' in position_name:
        regular_position_type = 'forward'
    elif 'midfield' in position_name or 'wing' in position_name:
        regular_position_type = 'midfield'
    elif 'back' in position_name or 'goalkeeper' in position_name:
        regular_position_type = 'defender'

    shot_taker_type.append(regular_position_type)

    technique = 'None'
    if 'technique' in shot['shot']:
        technique = shot['shot']['technique']['name']
    shot_technique.append(technique)

    if 'one_on_one' in shot['shot']:
        one_on_one.append(1)
    else:
        one_on_one.append(0)

    shot_body_part.append(shot['shot']['body_part']['name'])

    shot_location = shot['location']
    curr_dist = ((shot_location[0] - CENTRE_OF_GOAL[0])**2 + (shot_location[1] - CENTRE_OF_GOAL[1])**2)**0.5
    distance_to_goal.append(curr_dist)

    angle1 = atan2(LEFT_OF_GOAL[1] - shot_location[1], LEFT_OF_GOAL[0] - shot_location[0])
    angle2 = atan2(RIGHT_OF_GOAL[1] - shot_location[1], RIGHT_OF_GOAL[0] - shot_location[0])
    angle_to_goal.append(abs(min(angle1, angle2)))

    between_goal_count = 0

    players_within_1_count = 0
    players_within_5_count = 0
    players_within_10_count = 0
    opponents_within_5_count = 0
    teammates_within_5_count = 0

    for player in shot['shot']['freeze_frame']:
        curr_loc = player['location']
        between_goal_count += player_between_goal(shot_location, curr_loc) 

        if player['position']['name'] == 'Goalkeeper' and not player['teammate']:
            goalkeeper_in_the_way.append(player_between_goal(shot_location, curr_loc))

        dist_to_shot = ((curr_loc[0] - shot_location[0])**2 + (curr_loc[1] - shot_location[1])**2)**0.5
        if dist_to_shot < 1:
            players_within_1_count += 1
        if dist_to_shot < 5:
            players_within_5_count += 1
            if not player['teammate']:
                opponents_within_5_count += 1
            else:
                teammates_within_5_count += 1
        if dist_to_shot < 10:
            players_within_10_count += 1

    players_within_1.append(players_within_1_count)
    players_within_5.append(players_within_5_count)
    players_within_10.append(players_within_10_count)
    opponents_within_5.append(opponents_within_5_count)
    teammates_within_5.append(teammates_within_5_count)

    if len(goalkeeper_in_the_way) < len(players_within_1):
        goalkeeper_in_the_way.append(0)

    players_between_goal.append(between_goal_count)
    play_pattern.append(shot['shot']['type']['name'])
    duration.append(shot['duration'])

    if shot['shot']['outcome']['name'] == "Goal":
        goal.append(1)
    else:
        goal.append(0)


df = pd.DataFrame({
    'distant_to_goal': distance_to_goal,
    'play_pattern': play_pattern,
    'duration': duration,
    'angle_to_goal': angle_to_goal,
    'players_between_goal': players_between_goal,
    'players_within_1': players_within_1,
    'players_within_5': players_within_5,
    'players_within_10': players_within_10,
    'opponents_within_5': opponents_within_5,
    'teammates_within_5': teammates_within_5,
    'shot_taker_type': shot_taker_type,
    'shot_technique': shot_technique,
    'shot_body_part': shot_body_part,
    'goalkeeper_in_the_way': goalkeeper_in_the_way,
    'one_on_one': one_on_one,
    'goal': goal,
})

df.to_csv('shots.csv', index=True, header=True, index_label='shot_num')

key_pass_ids = []
for shot in data:
    if 'key_pass_id' in shot['shot']:
        key_pass_ids.append(shot['shot']['key_pass_id'])
related_df = pd.DataFrame({
    'key_pass_ids': key_pass_ids,
})
related_df.to_csv('key_pass_ids.csv', index=True, header=True, index_label='related_event_num')

print(df.describe(include='all'))