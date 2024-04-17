from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle5 as pickle
import pandas as pd
from collections import defaultdict
import json
import random
import torchvision
import re

import torch.nn.functional as F

class OURDS_CLIP_DataLoader(Dataset):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            csv_path,
            json_path,
            video_feature,
            bbx_feature,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            split_type="",
            split_task = None,
            mask_prob=0.15,
            use_real_name = False,
            is_pretraining = False,
            num_samples = 0, # if 0, use all samples, if > num_samples, use num_samples samples if < num_samples, use all samples
            only_players = True,
            use_random_embeddings = False,
            player_embedding_order = None,
            use_BBX_features = False,
            player_embedding="CLIP",
            max_rand_players = 0,
            action_convert_dict = None,
            
    ):
        self.data_stats = json.load(open('/home/karolwojtulewicz/code/NSVA/data/data_stats.json','r'))
        self.raw_frames_path = '/4TBSSD_permanent/NSVA/downscaled_frames'
        self.only_players = only_players
        self.is_pretraining = is_pretraining
        self.playerIDdict = None
        self.use_BBX_features = use_BBX_features
        self.max_rand_players = max_rand_players
        self.action_convert_dict = action_convert_dict
        if use_real_name:
            self.playerIDdict = json.load(open('data/player_info_dict.json','r'))
        self.mask_prob = mask_prob
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.feature_dict = video_feature
        self.bbx_feature_dict = bbx_feature
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.split_task = split_task
        self.player_embedding = player_embedding
        
        if use_random_embeddings:
            self.playerEmbeddingsDict = pickle.load(open('data/all_players_Random.pickle','rb'))
        else:
            if player_embedding == "CLIP":
                self.playerEmbeddingsDict = pickle.load(open('data/all_players_CLIP.pickle','rb'))
            elif player_embedding == "BERT":
                self.playerEmbeddingsDict = pickle.load(open('data/all_players_BERT.pickle','rb'))
            elif player_embedding == "BERT-Stat":
                self.playerEmbeddingsDict = pickle.load(open('data/all_players_BERT-Stat.pickle','rb'))
            else:
                raise NotImplementedError
        try:
            self.videoid2gameid_eventid = json.load(open('data/videoid2gameid_eventid.json','r'))
        except:
            print("videoid2gameid_eventid.json not found")
        
        self.player_embedding_order = player_embedding_order
        if "lineup" in player_embedding_order:
            with open('data/play_lineups.pickle', 'rb') as f:
                self.video_lineups = pickle.load(f)
        if "possession" in player_embedding_order:
            self.actionID_to_Name = {"action{}".format(index):value for index,value in enumerate(json.load(open('/home/karolwojtulewicz/code/NSVA/data/action_list.json','r')))}
            self.video_lineups = pickle.load(open('data/play_lineups.pickle', 'rb'))
            self.co_occurence_dict = self.data_stats["action_sequence_count"]
                
        self.feature_size = video_feature['video0'].shape[1]
        self.feature_size_bbx = 768
        self.multibbxs = True
        self.all_players = json.load(open('/home/karolwojtulewicz/code/NSVA/data/unique_players.json','r'))
        # remake all_players to be a dictionary
        self.all_players = {player:idx for idx,player in enumerate(self.all_players)}
           
            
        assert split_type in ["train", "val", "test"]

        prefixes = ['<T1>_','<T2>_','<T3>_','<T4>_']
        prefixes_in_use = [prefixes[s_id] for s_id,s_t in enumerate(self.split_task) if s_t ==1]
        sentences_in_use = []
        for sen in self.data['sentences']:
            use_sentence = False
            for p_i_u in prefixes_in_use:
                if p_i_u in sen['answer']:
                    use_sentence = True
                    break

            if use_sentence:
                sentences_in_use.append(sen)
        
        

        #self.data['sentences'] = sentences_in_use
        # only for temporary use, need to split dataset by game instead of video clip
        # train_nn = 35721
        # valid_nn = 4465
        # test_nn = 4465
        split_dict = json.load(open('{}/data/split2video_id_after_videos_combination.json'.format(os.environ["DIR_PATH"]), 'r'))
        self.videoid2split = {y:x for x in split_dict.keys() for y in split_dict[x]}
        #split_dict = {"train": video_ids[:train_nn], "val": video_ids[train_nn:train_nn + valid_nn], "test": video_ids[train_nn+valid_nn:]}
        choiced_video_ids = split_dict[split_type]
        choiced_video_ids = {vid:1 for vid in choiced_video_ids}

        self.sample_len = 0
        self.sentences_dict = {}
        self.video_sentences_dict = defaultdict(list)
        if split_type == "train":  # expand all sentence to train
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    #self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
                    self.video_sentences_dict[itm['video_id']].append(
                        {
                         "question": itm['question'],   
                         "answer":itm['answer'],
                         
                        }
                        )
                    """Convert the action to different label level"""
                    if '<T2>' in itm['caption'] and self.action_convert_dict is not None:
                        new_caption = itm['caption'].split('_')[-1].split(' ')
                        converted_caption = [self.action_convert_dict[x] for x in new_caption]
                        itm['caption'] = '<T2>_' + ' '.join(converted_caption)
                        itm['answer'] = '<T2>_' + ' '.join(converted_caption)
                        
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
                        
            all_train = [id for id, task in enumerate(self.split_task) if task ==1] 

            for vidx,obj in enumerate(self.data["sentences"]):
                #self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0])
                if obj["video_id"] not in choiced_video_ids:
                    continue
                if any("<T{}>_".format(value + 1) in obj["answer"] for value in all_train):
                    self.sentences_dict[len(self.sentences_dict)] = (obj["video_id"], obj)
                # self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid])
        elif split_type == "val" or split_type == "test":
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    # if itm['video_id'] == 'video34280':
                    #     print()
                    if prefixes_in_use[0] in itm['answer']:
                        self.video_sentences_dict[itm['video_id']].append(itm['answer'])
                    """Convert the action to different label level"""
                    if '<T2>' in itm['caption'] and self.action_convert_dict is not None:
                        new_caption = itm['caption'].split('_')[-1].split(' ')
                        converted_caption = [self.action_convert_dict[x] for x in new_caption]
                        itm['caption'] = '<T2>_' + ' '.join(converted_caption)
                        itm['answer'] = '<T2>_' + ' '.join(converted_caption)

                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])

            all_train = [id for id, task in enumerate(self.split_task) if task ==1] 

            for vidx,obj in enumerate(self.data["sentences"]):
                if obj["video_id"] not in choiced_video_ids:
                    continue
                #self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0])
                if any("<T{}>_".format(value + 1) in obj["answer"] for value in all_train):
                    #self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0])
                    vid = obj["video_id"]
                    self.sentences_dict[len(self.sentences_dict)] = (vid, obj)
        else:
            raise NotImplementedError

        if num_samples > 0:
            num_samples = min(num_samples, len(self.sentences_dict))
            # Sample keys from the dictionary
            sampled_keys = random.sample(list(self.sentences_dict.keys()), num_samples)
            # Rebuild the dictionary with a new index as the key
            self.sentences_dict = {new_index: self.sentences_dict[old_key] for new_index, old_key in enumerate(sampled_keys)}
        self.sample_len = len(self.sentences_dict)

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, answer, question):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=int)
        pairs_mask = np.zeros((k, self.max_words), dtype=int)
        pairs_segment = np.zeros((k, self.max_words), dtype=int)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=int)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=int)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=int)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=int)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=int)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(question)
            words = ["[CLS]"] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]

            # Mask Language Model <-----
            token_labels = []
            masked_tokens = words.copy()
            for token_id, token in enumerate(masked_tokens):
                if token_id == 0 or token_id == len(masked_tokens) - 1:
                    token_labels.append(-1)
                    continue
                prob = random.random()
                # mask token with 15% probability
                if prob < self.mask_prob:
                    prob /= self.mask_prob
                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        masked_tokens[token_id] = "[MASK]"
                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        masked_tokens[token_id] = random.choice(list(self.tokenizer.vocab.items()))[0]
                    # -> rest 10% randomly keep current token
                    # append current token to output (we will predict these later)
                    try:
                        token_labels.append(self.tokenizer.vocab[token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        token_labels.append(self.tokenizer.vocab["[UNK]"])
                        # print("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
                else:
                    # no masking token (will be ignored by loss function later)
                    token_labels.append(-1)
            # -----> Mask Language Model

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            masked_token_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                masked_token_ids.append(0)
                token_labels.append(-1)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words
            assert len(masked_token_ids) == self.max_words
            assert len(token_labels) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
            pairs_masked_text[i] = np.array(masked_token_ids)
            pairs_token_labels[i] = np.array(token_labels)

            # For generate captions
            task_type = -1
            if answer is not None:
                if '<T1>_' in answer and self.split_task[0] ==1:
                    task_type = 0
                    answer = answer[5:]
                elif '<T2>_' in answer and self.split_task[1] ==1:
                    task_type = 1
                    answer = answer[5:]
                elif '<T3>_' in answer and self.split_task[2] ==1:
                    task_type = 2
                    answer = answer[5:]
                elif '<T4>_' in answer and self.split_task[3] ==1:
                    task_type = 3
                    answer = answer[5:]
                else:
                    assert task_type!=-1
                caption_words = self.tokenizer.tokenize(answer)
            else:
                assert 1==0
                caption_words = self._get_single_text(video_id)
            
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = ["[CLS]"] + caption_words
            output_caption_words = caption_words + ["[SEP]"]

            # For generate captions
            input_answer_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_answer_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_answer_ids)
            while len(input_answer_ids) < self.max_words:
                input_answer_ids.append(0)
                output_answer_ids.append(0)
                decoder_mask.append(0)
            assert len(input_answer_ids) == self.max_words
            assert len(output_answer_ids) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_answer_ids)
            pairs_output_caption_ids[i] = np.array(output_answer_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)
        assert task_type!=-1
        return pairs_text, pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, choice_video_ids, task_type

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_video(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=int)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, self.feature_size))
        for i, video_id in enumerate(choice_video_ids):

            video_slice = self.feature_dict[video_id]
            
            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                if len(video_slice.shape) == 1:
                    print(self.videoid2gameid_eventid[video_id])
                    continue
                
                video[i][:slice_shape[0]] = video_slice

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # Mask Frame Model <-----
        video_labels_index = [[] for _ in range(len(choice_video_ids))]
        masked_video = video.copy()
        for i, video_pair_ in enumerate(masked_video):
            for j, _ in enumerate(video_pair_):
                if j < max_video_length[i]:
                    prob = random.random()
                    # mask token with x% probability
                    if prob < self.mask_prob:
                        masked_video[i][j] = [0.] * video.shape[-1]
                        video_labels_index[i].append(j)
                    else:
                        video_labels_index[i].append(-1)
                else:
                    video_labels_index[i].append(-1)
        video_labels_index = np.array(video_labels_index, dtype=int)
        # -----> Mask Frame Model

        return video, video_mask, masked_video, video_labels_index

    def _get_bbx(self, choice_video_ids):
        ## feature size for BBX is  
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=int)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, self.feature_size_bbx), dtype=np.float)
        for i, video_id in enumerate(choice_video_ids):

            if video_id not in self.bbx_feature_dict.keys():
                video_slice = video
                return video, video_mask, None, None
            else:
                video_slice = self.bbx_feature_dict[video_id]
            # else:
            #     gameid_eventid = self.videoid2gameid_eventid[video_id]
            #     # gameid = gameid_eventid.split('-')[0]
            #     # eventid = gameid_eventid.split('-')[1]
            #     video_slice = extracting_features(self.video_path+'/'+self.videoid2split[video_id]+'/'+gameid_eventid+'.mp4')

            # print("########{}".format(video_slice.shape))
            # if self.feature_size == 1024:
            #     video_slice = np.transpose(video_slice)
            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                if len(video_slice.shape) == 1:
                    print(self.videoid2gameid_eventid[video_id])
                    continue
                # print("!!!!!!!!!!! {}, {}".format(video[i][:slice_shape[0]].shape, video_slice.shape))
                video[i][:slice_shape[0]] = video_slice

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # Mask Frame Model <-----
        video_labels_index = [[] for _ in range(len(choice_video_ids))]
        masked_video = video.copy()
        for i, video_pair_ in enumerate(masked_video):
            for j, _ in enumerate(video_pair_):
                if j < max_video_length[i]:
                    prob = random.random()
                    # mask token with 15% probability
                    if prob < self.mask_prob:
                        masked_video[i][j] = [0.] * video.shape[-1]
                        video_labels_index[i].append(j)
                    else:
                        video_labels_index[i].append(-1)
                else:
                    video_labels_index[i].append(-1)
        video_labels_index = np.array(video_labels_index, dtype=int)
        # -----> Mask Frame Model

        return video, video_mask, masked_video, video_labels_index
        
    def _get_Player_Embedding(self, players):
        return_embeddings = None
        for player in players:
            if return_embeddings is None:
                return_embeddings = self.playerEmbeddingsDict[player]
            else:
                return_embeddings = torch.cat((return_embeddings, self.playerEmbeddingsDict[player.replace(",","")]), 0)
        return return_embeddings if return_embeddings is not None else torch.zeros(1,768)
        
    def _get_multi_bbxs(self, choice_video_ids, players):
        ## feature size for BBX is
        # if len(self.bbx_feature_dict['video10981'].shape) ==2:
        #     video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=int)
        # else:
        #     video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=int)
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=int)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids),self.bbx_feature_dict['video11030'].shape[0], self.max_frames, self.feature_size_bbx), dtype=np.float32)
        for i, video_id in enumerate(choice_video_ids):

            if video_id not in self.bbx_feature_dict.keys():
                video_slice = video
                return video, video_mask, None, None
            else:
                video_slice = self.bbx_feature_dict[video_id]
                clipEmbeddings = self._get_Player_Embedding(players)
                clipEmbeddings = clipEmbeddings.unsqueeze(0).expand(video_slice.shape[0],-1,-1)
                if self.use_BBX_features:
                    video_slice = np.concatenate((clipEmbeddings.detach().numpy(),video_slice), 1)
                else:
                    video_slice = clipEmbeddings.detach().numpy()
            # else:
            #     gameid_eventid = self.videoid2gameid_eventid[video_id]
            #     # gameid = gameid_eventid.split('-')[0]
            #     # eventid = gameid_eventid.split('-')[1]
            #     video_slice = extracting_features(self.video_path+'/'+self.videoid2split[video_id]+'/'+gameid_eventid+'.mp4')

            # print("########{}".format(video_slice.shape))
            # if self.feature_size == 1024:
            #     video_slice = np.transpose(video_slice)
            if self.max_frames < video_slice.shape[1]:
                video_slice = video_slice[:,:self.max_frames,:]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[1] else slice_shape[1]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                if len(video_slice.shape) == 1:
                    print(self.videoid2gameid_eventid[video_id])
                    continue
                # print("!!!!!!!!!!! {}, {}".format(video[i][:slice_shape[0]].shape, video_slice.shape))
                for j in range(slice_shape[0]):
                    video[i][j][:slice_shape[1]] = video_slice[j]

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # Mask Frame Model <-----
        video_labels_index = [[] for _ in range(len(choice_video_ids))]
        masked_video = video.copy()
        for i, video_pair_ in enumerate(masked_video):
            for j, _ in enumerate(video_pair_[0]):
                if j < max_video_length[i]:
                    prob = random.random()
                    # mask token with 15% probability
                    if prob < self.mask_prob:
                        for k in range(video_pair_.shape[0]):
                            masked_video[i][k][j] = [0.] * video.shape[-1]
                        video_labels_index[i].append(j)
                    else:
                        video_labels_index[i].append(-1)
                else:
                    video_labels_index[i].append(-1)
        video_labels_index = np.array(video_labels_index, dtype=int)
        # -----> Mask Frame Model

        return video, video_mask, masked_video, video_labels_index
    
    def simulate_ball_possession(self,gt_players, random_players, actions, co_occurrence_actions):
        
        # Ensure there's enough total players
        total_players_needed = random.randint(len(gt_players), len(gt_players)+self.max_rand_players)
        random_players += random_players
        total_players = random.sample(random_players, total_players_needed)
        
        # Check for "Made" condition and co-occurrence in the last actions
        action_name = self.actionID_to_Name[actions[-1]]
        made_condition = "Made" in action_name  and len(gt_players) >1 and random.random() < 0.5
        co_occurrence_condition = False
        if len(actions) > 1:
            co_occurrence_condition = " ".join(actions[-2:]) in co_occurrence_actions

        # print("M:", made_condition, "C:", co_occurrence_condition, "Act len:", len(actions), "GT len:", len(gt_players))
        # Initialize the augmented player list
        augmented_players = []
        augmented_players.extend(total_players)
        random_indecies = None
        while random_indecies is None:
            tmp = [random.randint(0, len(augmented_players)-1) for _ in range(len(gt_players)-1)]
            if len(list(set(tmp))) == len(tmp):
                random_indecies = tmp
                random_indecies.sort()
        if made_condition or co_occurrence_condition:
            # Preserve the last GT player for all scenarios
            preserved_last_gt_players = gt_players[-2:]
            
            # Randomly insert the remaining GT players
            
            
                    
            for gt_player in gt_players[:-1]:  # Excluding the last player for now
                insert_pos = random_indecies.pop(0)
                augmented_players.insert(insert_pos, gt_player)
            
            # Add the random players
            
            augmented_players.extend(preserved_last_gt_players)
        else:
            
            for gt_player in gt_players[:-1]:  # Excluding the last player for now
                insert_pos = random_indecies.pop(0)
                augmented_players.insert(insert_pos, gt_player)
                    
            # Ensure the last GT player is at the end
            if gt_players:
                augmented_players.append(gt_players[-1])

        return augmented_players


    def getLineupData(self, sentence):
        
        gt_players = sentence["question"].split(" ")
        actions = sentence["actions"].split(" ")
        video_id = sentence["video_id"]
        
        if video_id not in self.video_lineups:
            # get closest video_id
            numbers = sorted(int(key.replace('video', '')) for key in self.video_lineups.keys())
            video_id = 'video{}'.format(min(numbers, key=lambda x:abs(x-int(video_id.replace('video', '')))))
            
        lineups = self.video_lineups[video_id]
        home_lineup, away_lineup = list(lineups["home_lineup"]), list(lineups["away_lineup"])
        
        def filter_func(player):
            if player in gt_players:
                return False
            return True
        lineup = home_lineup + away_lineup
        filtered_lineup = list(filter(filter_func,lineup))[:7]
        return gt_players, filtered_lineup, self.co_occurence_dict, actions, video_id

    def __getitem__(self, idx):
        video_id, qa_object = self.sentences_dict[idx]
        
        
        video, video_mask, masked_video, video_labels_index = self._get_video([video_id])
        
            
            

        #bbx, bbx_mask, masked_bbx, bbx_labels_index = self._get_bbx([video_id])


        question = qa_object["question"]
        answer = qa_object["answer"]
        player_IDs = []
        if "PLAYER" in question:
            for action in question.replace("<T3>_", "").split():
                action = action.strip()
                player_IDs.append(action)
        if self.playerIDdict and  ("PLAYER" in question):
            prefix = question.replace("<T3>_", "").split()[0]
            for action in question.replace("<T3>_", "").split():
                action = action.strip()
                player_IDs.append(action)
                playerInfo = self.playerIDdict[action.replace("PLAYER", "").strip()]
                answer = answer.replace(action, " " + playerInfo["first_name"] + " " + playerInfo["last_name"] + " ")
                question = question.replace(action, " " + playerInfo["first_name"] + " " + playerInfo["last_name"])
            question = prefix + " " + question
            
        if  "lineup" in self.player_embedding_order:
            if "_" in question:
                prefix, players = question.split("_")
            else:
                prefix = "<T1>"
                players = question.replace(":", "").replace(";", "").replace("side", "")
            # players are a string of players separated by a space
            players = players.split()
            
            # get the lineup for the videoa
            try:
                lineup = self.video_lineups[video_id]
            except:
                lineup = {"home_lineup": [], "away_lineup": []}
            all_lineup = list(lineup["home_lineup"]) + list(lineup["away_lineup"])
            # now we have list of all players in the lineup
            # check if the players in the question are in the lineup
            # if not, replace them with random players from the lineup
            
            # remove all players from all_lineup that are in players
            for action in players:
                if action in all_lineup:
                    all_lineup.remove(action)
            
            if len(all_lineup) + len(players) > 10:
                all_lineup = all_lineup[:10-len(players)] + players
            else:
                all_lineup = all_lineup + players
            
            # now all_lineup has 10 players
            # shuffle the list
            if self.player_embedding_order != "lineup-ordered":
                random.shuffle(all_lineup)
            
            question = prefix + "_ " + " ".join(all_lineup)
            player_IDs = all_lineup
        if "possession" in self.player_embedding_order:
            gt_players, random_players, co_occurrence_actions, actions, video_id = self.getLineupData(qa_object)
            
            question = "<T3>_"
            player_IDs = []
            if gt_players[0]  != "":
                player_IDs = self.simulate_ball_possession(gt_players, random_players, actions, co_occurrence_actions)
                player_IDs = [player.replace(":", "").replace("side", "").replace(",","").replace(".","").strip() for player in player_IDs if player != ""]
                question = "<T3>_".join(player_IDs)
            # else:
            #     print("No players in possession")
            
        
        if self.only_players:
            question = answer.split("_")[0]+"_ "+ question.split("_")[1]
        
            
        if self.multibbxs:
            bbx, bbx_mask, masked_bbx, bbx_labels_index = self._get_multi_bbxs([video_id], player_IDs)
            if bbx is None or bbx_mask is None or masked_bbx is None or bbx_labels_index is None:
                masked_bbx = torch.zeros(1,2, self.max_frames, self.feature_size_bbx)
                bbx_labels_index = torch.zeros(1, self.max_frames)

        else:
            bbx, bbx_mask, masked_bbx, bbx_labels_index = self._get_bbx([video_id])
        
        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, \
        pairs_input_caption_ids, pairs_decoder_mask, \
        pairs_output_caption_ids, choice_video_ids, task_type = self._get_text(video_id, answer, question)
        
        
        

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,task_type, torch.tensor(bbx), torch.tensor(bbx_mask), torch.tensor(masked_bbx), torch.tensor(bbx_labels_index)
