import torch
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

model_dir = "/home/share/model_weight/qwen/Qwen2.5-VL-7B-Instruct/" 

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation = "flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_dir)


con = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video":"/home/wmk/code/data/VideoDetailCaption/Test_Videos/v__B7rGFDRIww.mp4",
                    "max_pixels": 224*224,
                    "min_pixels": 224*224,
                    "fps": 0.5,
                },
                {"type": "text", "text": "Describe what happen in the video?"},
            ],
        }
    ]
   
text = processor.apply_chat_template(con, tokenize=False, add_generation_prompt=True)

image_inputs, video_inputs, video_kwargs = process_vision_info(con, return_video_kwargs=True)
inputs = processor(text=text, videos=video_inputs,return_tensors="pt",**video_kwargs).to(model.device, torch.float16)
print("INFO: Input length:", inputs['input_ids'].shape[1])


generated_ids = model.generate(**inputs, max_new_tokens=128)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)