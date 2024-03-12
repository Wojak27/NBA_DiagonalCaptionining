import torch
import pickle
import json
from transformers import AutoTokenizer, CLIPTextModelWithProjection, BertModel, BertTokenizer

# Define your device based on CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and tokenizer
clip_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Load BERT model and tokenizer
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load player data
with open('/home/karolwojtulewicz/code/NSVA/all_unique_players.pickle', "rb") as f:
  all_players = pickle.load(f)

with open("/home/karolwojtulewicz/code/NSVA/data/player_info_dict.json") as f:
    detailed_player_info = json.load(f)
  
# Convert to list
all_players = [player.replace("PLAYER","") for player in list(all_players.keys())]

pickle_dict = {}
embedding_type = "BERT-Stat"  # Change this to "CLIP", "BERT", or "random"

for player in all_players:
    print(player)
    if embedding_type == "CLIP":
        player_name = detailed_player_info[player]["first_name"] + " " + detailed_player_info[player]["last_name"]
        inputs = clip_tokenizer([player_name], padding=True, return_tensors="pt").to(device)
        outputs = clip_model(**inputs)
        text_embeds = outputs.text_embeds
    elif embedding_type == "BERT":
        player_name = detailed_player_info[player]["first_name"] + " " + detailed_player_info[player]["last_name"]
        inputs = bert_tokenizer([player_name], padding=True, return_tensors="pt", truncation=True).to(device)
        outputs = bert_model(**inputs)
        # Average the embeddings across the token dimension
        text_embeds = outputs.last_hidden_state.mean(dim=1)
    elif embedding_type == "BERT-Stat":
        player_name = detailed_player_info[player]["first_name"] + " " + detailed_player_info[player]["last_name"]
        stats = " ".join([str(key) + ": " +str(detailed_player_info[player][key]) + "; " for key in detailed_player_info[player] if key not in ["first_name", "last_name"]])
        player_name += "; " + stats
        inputs = bert_tokenizer([player_name], padding=True, return_tensors="pt", truncation=True).to(device)
        outputs = bert_model(**inputs)
        # Average the embeddings across the token dimension
        text_embeds = outputs.last_hidden_state.mean(dim=1)
    else:  # Random embeddings
        text_embeds = torch.randn(1, 768).to(device)

    pickle_dict["PLAYER{}".format(player)] = text_embeds.cpu()

print("Dict length: ", len(pickle_dict))

# Save the embeddings to a pickle file
filename = f"data/all_players_{embedding_type}.pickle"
with open(filename, "wb") as f:
    pickle.dump(pickle_dict, f)
