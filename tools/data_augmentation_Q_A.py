import json

# Sample data structure (replace this with the actual data)
data = json.load(open('/home/karolwojtulewicz/code/NSVA/data/ourds_description_only.json', 'r'))

def extract_player_action(caption):
    caption = caption.replace("_", " ")
    parts = caption.split(" ")
    players = [part for part in parts if part.startswith('PLAYER')]
    actions = [part for part in parts if part.startswith('action')]
    return players, actions

def generate_questions_answers(data):
    questions = []
    answers = []
    print("Caption: {}".format(data[0]['caption']))
    # Player-Specific Questions
    first_player, _ = extract_player_action(data[0]['caption'])
    questions.append("Which player made the first action?")
    answers.append(first_player[0] if first_player else "No player")

    last_player, _ = extract_player_action(data[-1]['caption'])
    questions.append("Who performed the last action in the segment?")
    answers.append(last_player[0] if last_player else "No player")

    all_players = [player for item in data for player in extract_player_action(item['caption'])[0]]
    json.dump(all_players, open('all_players.json', 'w'))
    unique_players = set(all_players)
    questions.append("How many players players are involved in the actions?")
    answers.append(len(unique_players))

    most_frequent_player = max(set(all_players), key=all_players.count) if all_players else "No player"
    questions.append("Who performed the most actions?")
    answers.append(most_frequent_player)

    questions.append("Did any player perform both an offensive and a defensive action?")
    answers.append("Yes" if len(unique_players) < len(all_players) else "No")

    # Action-Specific Questions
    _, first_action = extract_player_action(data[0]['caption'])
    questions.append("What is the first action taken in the segment?")
    answers.append(first_action[0] if first_action else "No action")

    all_actions = [action for item in data for action in extract_player_action(item['caption'])[1]]
    unique_actions = set(all_actions)
    questions.append("How many different types of actions are there?")
    answers.append(len(unique_actions))

    questions.append("Is there any 'Defensive Rebound' action?")
    answers.append("Yes" if 'action135' in all_actions else "No")

    questions.append("What is the most frequent action?")
    most_frequent_action = max(set(all_actions), key=all_actions.count) if all_actions else "No action"
    answers.append(most_frequent_action)

    questions.append("What action immediately follows any '3PT Jump Shot'?")
    for i in range(len(data) - 1):
        _, actions = extract_player_action(data[i]['caption'])
        if 'action109' in actions:
            _, next_actions = extract_player_action(data[i + 1]['caption'])
            answer = next_actions[0] if next_actions else "No subsequent action"
            break
    else:
        answer = "No '3PT Jump Shot' action found"
    answers.append(answer)

    return questions, answers

questions, answers = generate_questions_answers(data["sentences"])

for q, a in zip(questions, answers):
    print(f"Q: {q}\nA: {a}\n")
