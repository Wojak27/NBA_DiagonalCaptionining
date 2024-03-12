from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from collections import defaultdict
import json
import torch

from torch.utils.data import Dataset
import numpy as np
import pickle
import random
import torch.nn.functional as F

class ActivityNet_Caption_DataLoader_Flow(Dataset):
    """ActivityNet Caption train dataset loader."""
    def __init__(
            self,
            annotations_path,
            v_features_path,
            a_features_path,
            json_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            split_type=""
    ):
        self.annotations = json.load(open(annotations_path, "r"))
        # The features are too big so we need to load them on the fly
        self.v_feature_dir = v_features_path
        self.a_feature_dir = a_features_path
        self.data = json.load(open(json_path, 'r'))
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        assert ".pkl" not in self.v_feature_dir and ".pickle" not in self.v_feature_dir
        assert ".pkl" not in self.a_feature_dir and ".pickle" not in self.a_feature_dir
        

        # in some datasets videos do not have a prefix "v_", so we need to check it
        try:
            self.v_feature_size = np.load(self.v_feature_dir+"/"+self.annotations[0]+".npy").shape[-1]
        except:
            self.v_feature_size = np.load(self.v_feature_dir+"/"+self.annotations[0].replace("v_", "")+".npy").shape[-1]
            
        try:
            self.a_feature_size = np.load(self.a_feature_dir+"/"+self.annotations[0]+".npy").shape[-1]
        except:
            self.a_feature_size = np.load(self.a_feature_dir+"/"+self.annotations[0].replace("v_", "")+"_bn.npy").shape[-1]

        assert split_type in ["train", "val", "test"]

        self.sample_len = 0
        self.sentences_dict = {}
        self.video_sentences_dict = defaultdict(list)
        if split_type == "train":
            for video_id, content in self.data.items():
                if video_id in self.annotations:
                    sentence =  "".join(content['sentences'])
                    self.sentences_dict[len(self.sentences_dict)] = (video_id, sentence)
                    self.video_sentences_dict[video_id].append(sentence)

        elif split_type in ["val", "test"]:
            for video_id, content in self.data.items():
                if video_id in self.annotations:
                    self.video_sentences_dict[video_id].extend(["".join(content['sentences'])])
            
            for vid in self.annotations:
                if vid in self.video_sentences_dict:
                    self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0])

        else:
            raise NotImplementedError

        self.sample_len = len(self.sentences_dict)

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.longlong)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.longlong)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.longlong)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.longlong)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.longlong)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.longlong)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.longlong)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.longlong)

        for i, video_id in enumerate(choice_video_ids):
            words = []
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
                if prob < 0.15:
                    prob /= 0.15
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
            if caption is not None:
                caption_words = self.tokenizer.tokenize(caption)
            else:
                caption_words = self._get_single_text(video_id)
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = ["[CLS]"] + caption_words
            output_caption_words = caption_words + ["[SEP]"]

            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_caption_ids)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(0)
                output_caption_ids.append(0)
                decoder_mask.append(0)
            assert len(input_caption_ids) == self.max_words
            assert len(output_caption_ids) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)

        return pairs_text, pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = " ".join(self.sentences[video_id][rind])
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_video(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.longlong)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, self.v_feature_size), dtype=np.longfloat)
        for i, video_id in enumerate(choice_video_ids):
            try:
                video_slice = np.load(self.v_feature_dir+"/"+video_id+".npy")
            except:
                video_slice = np.load(self.v_feature_dir+"/"+video_id.replace("v_", "")+".npy")

            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
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
                    if prob < 0.15:
                        masked_video[i][j] = [0.] * video.shape[-1]
                        video_labels_index[i].append(j)
                    else:
                        video_labels_index[i].append(-1)
                else:
                    video_labels_index[i].append(-1)
        video_labels_index = np.array(video_labels_index, dtype=np.longlong)
        # -----> Mask Frame Model

        return video, video_mask, masked_video, video_labels_index
    
    def interpolate_sequence(self, data, target_length=200):
        """
        Adjusted function to correctly interpolate the sequence to a target length 
        while preserving the embedding dimensions.

        :param data: Tensor of shape (seqlen, embedding_dim)
        :param target_length: Target length for interpolation
        :return: Interpolated tensor of shape (target_length, embedding_dim)
        """
        # Check if data is a numpy array and convert it to a PyTorch tensor
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        # Get the current sequence length and embedding dimension
        seqlen, embedding_dim = data.shape

        # Add a dummy batch dimension: shape becomes (1, seqlen, embedding_dim)
        data = data.unsqueeze(0)
        data = data.transpose(1, 2)

        # Interpolate along the sequence length dimension
        # size parameter is now a tuple with target_length as the sequence length
        # and embedding_dim remains the same
        resized_data = F.interpolate(data, size=(target_length, ), mode='linear', align_corners=False)

        # Remove the dummy batch dimension: shape becomes (target_length, embedding_dim)
        resized_data = resized_data.transpose(1, 2)
        resized_data = resized_data.squeeze(0)

        return resized_data
    
    def _get_audio(self, choice_audio_ids):
        audio_mask = np.zeros((len(choice_audio_ids), self.max_frames), dtype=np.longlong)
        max_audio_length = [0] * len(choice_audio_ids)

        audio = np.zeros((len(choice_audio_ids), self.max_frames, self.a_feature_size), dtype=np.longfloat)
        for i, audio_id in enumerate(choice_audio_ids):
            try:
                audio_slice = np.load(self.a_feature_dir+"/"+audio_id+"_bn.npy")
            except:
                try:
                    audio_slice = np.load(self.a_feature_dir+"/"+audio_id.replace("v_", "")+"_bn.npy")
                except:
                    print("Not found file for audio_id: {}".format(audio_id))
                    audio_slice = np.zeros((self.max_frames, self.a_feature_size), dtype=np.longfloat)

            if self.max_frames < audio_slice.shape[0]:
                audio_slice = self.interpolate_sequence(audio_slice, self.max_frames)

            slice_shape = audio_slice.shape
            max_audio_length[i] = max_audio_length[i] if max_audio_length[i] > slice_shape[0] else slice_shape[0]
            if len(audio_slice) < 1:
                print("audio_id: {}".format(audio_id))
            else:
                audio[i][:slice_shape[0]] = audio_slice

        for i, a_length in enumerate(max_audio_length):
            audio_mask[i][:a_length] = [1] * a_length

        # Mask Frame Model <-----
        audio_labels_index = [[] for _ in range(len(choice_audio_ids))]
        masked_audio = audio.copy()
        for i, audio_pair_ in enumerate(masked_audio):
            for j, _ in enumerate(audio_pair_):
                if j < max_audio_length[i]:
                    prob = random.random()
                    # mask token with 15% probability
                    if prob < 0.15:
                        masked_audio[i][j] = [0.] * audio.shape[-1]
                        audio_labels_index[i].append(j)
                    else:
                        audio_labels_index[i].append(-1)
                else:
                    audio_labels_index[i].append(-1)
        audio_labels_index = np.array(audio_labels_index, dtype=np.longlong)
        # -----> Mask Frame Model

        return audio, audio_mask, masked_audio, audio_labels_index

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]

        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, \
        pairs_input_caption_ids, pairs_decoder_mask, \
        pairs_output_caption_ids, choice_video_ids = self._get_text(video_id, caption)

        video, video_mask, masked_video, video_labels_index = self._get_video(choice_video_ids)
        audio, audio_mask, masked_audio, audio_labels_index = self._get_audio(choice_video_ids)
        # Cast to float32
        video = video.astype(np.float32)
        video_mask = video_mask.astype(np.float32)
        masked_video = masked_video.astype(np.float32)
        # Cast to float32
        audio = audio.astype(np.float32)
        audio_mask = audio_mask.astype(np.float32)
        masked_audio = masked_audio.astype(np.float32)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, \
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,2, audio, audio_mask


if __name__ == "__main__":
    # load dataset and return a batch
    import sys
    sys.path.append("/home/karolwojtulewicz/code/NSVA")
    from modules.tokenization import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    
    dataset = ActivityNet_Caption_DataLoader_Flow(
        annotations_path="/4TBSSD_permanent/collaborative-experts/data/anet_caption/train_ids.json",
        v_features_path="/4TBSSD_permanent/collaborative-experts/data/activity_net/env_c3d",
        a_features_path="/4TBSSD_permanent/collaborative-experts/data/activity_net/training",
        json_path="/4TBSSD_permanent/collaborative-experts/data/anet_caption/train.json",
        max_words=30,
        feature_framerate=1,
        tokenizer=tokenizer,
        max_frames=30,
        split_type="train",
    )
    
    print(len(dataset))
    print(dataset[1000])
