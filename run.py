import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
from tqdm import tqdm
import json
import argparse
from utils.choices import mc_sim_7b_63, chain
from utils.utils import *
from utils.decoding import *
#/home/wmk/code/model_weight/Qwen2.5-VL-32B-Instruct
def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--model_type', type=str, default='qwen2_5_vl', help='Model type: qwen2_5_vl')
    parser.add_argument('--target_model_path', type=str, default='/home/wmk/code/model_weight/Qwen2.5-VL-32B-Instruct', help='target model')
    parser.add_argument('--draft_model_path', type=str, default='/home/share/model_weight/qwen/Qwen2.5-VL-7B-Instruct/', help='draft model')
    parser.add_argument('--verbose', action='store_true', help='verbose')

    parser.add_argument('--task', type=str, default='VideoDetailCaption', choices=['VideoDetailCaption', 'MVBench', 'MVLU', 'LongVideoBench', 'MMBench'], help='dataset')
    parser.add_argument('--data_path', type=str,default='/home/wmk/code/data/VideoDetailCaption', help='Path to the data directory')
    parser.add_argument('--data_num', type=int, default=100, help='Number of data samples to load')
    parser.add_argument('--evaluation_num', type=int, default=1,help='Number of evaluation samples')
    parser.add_argument('--frame_num', type=int, default=168, help='Number of frames per video')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save results.')

    parser.add_argument('--temp', type=float, default=0.6, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='top_p')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Maximum number of new tokens to generate')

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":

    args = parse_arguments()

    model_type = args.model_type
    target_model_path = args.target_model_path
    draft_model_path = args.draft_model_path
    data_path = args.data_path
    task = args.task
    max_new_tokens = args.max_new_tokens
    evaluation_num = args.evaluation_num
    top_k = -1
    top_p = args.top_p
    temperature = args.temp
    frame_num=args.frame_num

    target_model, draft_model, processor = load_model(model_type, target_model_path, draft_model_path)
    data_video = load_data(task, args.data_num, data_path)
    if args.save_path is None:
        save_path = f"results/{model_type}_{task}"
    else:
        save_path = args.save_path

    # Run evaluation
    target_model.eval()
    draft_model.eval()

    results = {}

    results['Autoregressive_decoding'] = []

    results['speculative_decoding'] = []
    results['speculative_decoding_accept_length'] = []

    results['TriVLM_decode'] = []
    results['TriVLM_accept_length'] = []

    for i in tqdm(range(evaluation_num)):
        data_instance = data_video[i]
        inputs, video_inputs = decode_video(processor, task, data_instance,frame_num = frame_num, model_type = model_type, data_path=data_path)
        if inputs == None:
            continue

        # output_ar = Autoregressive(inputs, video_inputs, target_model ,max_new_tokens=max_new_tokens, top_k=top_k, top_p=top_p, temperature=temperature)
        # print("\n")
        # print("-------Autoregressive Decoding-------")
        # #print("Inference Time:", output_ar['inference_time'])
        # print("Decoding Time:", output_ar['decoding_time'])
        # output_text = processor.batch_decode(output_ar['output_ids'], skip_special_tokens=True)[0]
        # print("Output:")
        # print(output_text)
        # print("\n")
        # results['Autoregressive_decoding'].append(output_ar['decoding_time'])

        # output_sd = speculative_decoding(
        #         inputs,
        #         video_inputs,
        #         target_model,
        #         draft_model,
        #         processor,
        #         max_new_tokens=max_new_tokens,
        #         tree_choices=mc_sim_7b_63,
        #         top_k=top_k,
        #         top_p=top_p,
        #         temperature=temperature,
        # )
        # print("\n")
        # print("-------Naive Speculative Decoding (with tree attn)-------")
        # #print("Inference Time:", output_sd['inference_time'])
        # print("Decoding Time:", output_sd['decoding_time'])
        # print("Average Accept Length:", output_sd["mean_accept_length"].item())
        # output_text = processor.batch_decode(output_sd['output_ids'], skip_special_tokens=True)[0]
        # print("Output:")
        # print(output_text)
        # print("\n")
        # results['speculative_decoding'].append(output_sd['decoding_time'])
        # results['speculative_decoding_accept_length'].append(output_sd["mean_accept_length"])

        output_specvlm = sparse_speculative_decoding(
            inputs,
            video_inputs,
            target_model,
            draft_model,
            processor,
            max_new_tokens=max_new_tokens,
            tree_choices=mc_sim_7b_63,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        print("\n")
        print("-------TriVLM-------")
        # print("Inference Time:", output_specvlm['inference_time'])
        print("Decoding Time:", output_specvlm['decoding_time'])
        print("Average Accept Length:", output_specvlm["mean_accept_length"].item())
        output_text = processor.batch_decode(output_specvlm['output_ids'], skip_special_tokens=True)[0]
        print("Output:")
        print(output_text)
        print("\n")
        results['TriVLM_decode'].append(output_specvlm['decoding_time'])
        results['TriVLM_accept_length'].append(output_specvlm["mean_accept_length"])

        # if save_path is not None:
        #     print("\n")
        #     print("-------Average Results-------")
        #     print("Autoregressive Decoding Time:", sum(results['Autoregressive_decoding'])/len(results['Autoregressive_decoding']))
        #     print("\n")
        #     print("Naive SD Decoding Time:", sum(results['speculative_decoding'])/len(results['speculative_decoding']))
        #     print("Naive SD Average Accept Length:", (sum(results['speculative_decoding_accept_length'])/len(results['speculative_decoding_accept_length'])).item())
        #     print("\n")
        #     print("-------End-------")\

        #     metrics = {
        #     "Autoregressive Decoding Time": float(sum(results['Autoregressive_decoding'])/len(results['Autoregressive_decoding'])),
        #     "Naive SD": {
        #         "Decoding Time": float(sum(results['speculative_decoding'])/len(results['speculative_decoding'])),
        #         "Average Accept Length": float(sum(results['speculative_decoding_accept_length'])/len(results['speculative_decoding_accept_length']))
        #     },
        #     }

        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     with open(save_path, 'w') as f:
        #         json_str = json.dumps(metrics, indent=4)
        #         f.write(json_str + '\n\n')

    
