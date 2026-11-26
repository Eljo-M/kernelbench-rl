import json, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
path='/content/models/Qwen2.5-Coder-7B-Instruct'
quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
started=time.time()
model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,quantization_config=quant,torch_dtype=torch.float16,device_map={'':'cuda:0'})
tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
text=tokenizer.apply_chat_template([{'role':'user','content':'Write a Python function that adds two numbers. Return code only.'}],tokenize=False,add_generation_prompt=True)
inputs=tokenizer(text,return_tensors='pt').to('cuda')
with torch.no_grad():
    outputs=model.generate(**inputs,max_new_tokens=64,do_sample=False,pad_token_id=tokenizer.eos_token_id)
result={'model':'Qwen/Qwen2.5-Coder-7B-Instruct','quantization':'NF4 double quantized 4bit','torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'seconds':time.time()-started,'completion':tokenizer.decode(outputs[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True),'training_performed':False}
Path('/content/probe_7b.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
