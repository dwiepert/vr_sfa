import numpy as np
import copy
import warnings
from decord import VideoReader, cpu
import json
from tqdm import tqdm
import argparse
import os


# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--data_folder', type=str, default="./TemporalBench_local_data", help='Path to dataset (from Huggingface)')
parser.add_argument('--data_json', type=str, default="temporalbench_short_qa.json", help='which type ')
parser.add_argument('--ckpt_folder', type=str, default="lmms-lab", help='Folder to model checkpoints')
parser.add_argument('--model_name', type=str, default="llava-onevision-qwen2-7b-ov", help='Path to model checkpoints')
parser.add_argument("--output_folder", type=str, default="./outputs", help="Output directory of score files")
parser.add_argument("--nframes", type=int, default=1, help="Number of frames to sample.")

# Parse arguments
args = parser.parse_args()

output_dir = os.path.join(args.output_folder, args.data_json.split('.')[0])
nframes = args.nframes
os.makedirs(output_dir, exist_ok=True)


##################### Initilaize the model #####################

import torch
import torch.nn as nn
# OpenGVLab/VideoMAEv2-Large

# from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
# import numpy as np
# import torch


# config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-Large", trust_remote_code=True)
# # theoretically we already have the videos processed
# # processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-Large")
# model = AutoModel.from_pretrained('OpenGVLab/VideoMAEv2-Large', config=config, trust_remote_code=True)
# model.eval()

from transformers import T5Tokenizer, T5ForConditionalGeneration
text_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
text_encoder = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
print('hidden', text_encoder.config.d_model)

# feature_projector = nn.Linear() don't need feature projector bc feature dim and t5 hidden dim are both 768

# class VideoMAEEvalHead(nn.Module):
#     def __init__(self, video_feat_dim=768, text_feat_dim=768, hidden_dim=512):
#         super().__init__()
        

#     def score(self, video_feat, question, choices):
#         scores = []
#         for choice in choices:
#             print(choice)
#             inputs = self.text_tokenizer(question, choice, return_tensors="pt", truncation=True, padding=True)
#             print(inputs)

#             text_emb = self.text_encoder(**inputs).last_hidden_state[:, 0, :]  # CLS
#             fused = torch.cat([video_feat.unsqueeze(0), text_emb], dim=-1)
#             score = self.classifier(fused)  # shape [1, 1]
#             scores.append(score)
#         return torch.cat(scores, dim=1)  # shape [1, num_choices]



# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


##################### Get response #####################

# evaluator = VideoMAEEvalHead()

def get_response(video_path, nframes, question):
        # print(video_path)
        # video_features/short_video/FineGym/e3EsDlpNo0c_E_003078_003091_A_0000_0012.npz
        # short_video/FineGym/e3EsDlpNo0c_E_003078_003091_A_0000_0012.mp4
        base, _ = os.path.splitext(video_path)
        feature_path = "video_features/" + base + ".npz"
        
        # some videos got filtered out.
        try:
            # print("successfully loaded feature", feature_path)
            vid_feature = torch.from_numpy(np.load(feature_path)['arr_0']) # something, 768

            inputs = text_tokenizer(question['question'], return_tensors="pt")
            labels_A = text_tokenizer(f"{question['question']} A", return_tensors="pt").input_ids
            labels_B = text_tokenizer(f"{question['question']} B",  return_tensors="pt").input_ids

            with torch.no_grad():
                # print('pre token')
                tokens = text_encoder.encoder.embed_tokens(inputs.input_ids)
                # print('pre cat')
                lang_and_features = torch.cat([vid_feature.unsqueeze(0), tokens], dim=1)
                # print(lang_and_features.shape)
                # print('pre encoder')
                encoder_outputs = text_encoder.encoder(inputs_embeds=lang_and_features)

                # print('pre forward')

                output_A = text_encoder(encoder_outputs=encoder_outputs, decoder_input_ids=labels_A, labels=labels_A)
                output_B = text_encoder(encoder_outputs=encoder_outputs, decoder_input_ids=labels_B, labels=labels_B)

            output_loss_A = -output_A.loss.item()
            output_loss_B = -output_B.loss.item()
            prediction = "A"
            if output_loss_B > output_loss_A: prediction = "B" 

            if prediction == question["GT"]:
                print("correct", prediction)
            else:
                print("incorrect", prediction)
            return prediction


            # prepare conversation input
        #     conv_template = "qwen_2"
        #     prompt = f"{DEFAULT_IMAGE_TOKEN}\n" + question["question"] 
        # #   + "\nPlease only output one English character."

        #     input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        #     image_sizes = [frame.size for frame in video_frames]

        # Generate response
    #     cont = model.generate(
    #         input_ids,
    #         images=image_tensors,
    #         image_sizes=image_sizes,
    #         do_sample=False,
    #         temperature=0,
    #         max_new_tokens=4096,
    #         modalities=["video"],
    #     )
        # text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        
    #     reponse = text_outputs[0]


        except OSError:
            pass
        except Exception as e:
            print(e)
            # print("we skipped this file", feature_path) 

        return {}


with open(os.path.join(args.data_folder, args.data_json), 'r') as f:
    questions = json.load(f)
    



text_ans_file = open(os.path.join(output_dir, f"{args.model_name}-frame{nframes}.jsonl"), 'w')



for question in tqdm(questions):
    try:
        # Load and process video
        video_path = os.path.join(args.data_folder, question["video_name"])
        
        response = get_response(video_path, nframes, question)
        text_ans_file.write(json.dumps(dict(idx=question["idx"], response = response)) + '\n')
        text_ans_file.flush()
        
    except Exception as e:
        print(f"Error running video: {e}")
        continue

text_ans_file.close()