import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LlamaAI:

    def __init__(self):
        self.model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            device_map="auto",  # nếu dùng GPU
            low_cpu_mem_usage=True
        )

    def question(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(text)
