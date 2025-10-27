import re
import json
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LlamaAI:
    def __init__(self):
        print(torch.cuda.get_device_name(0))
        print(torch.version.cuda)
        torch.cuda.empty_cache()

        # self.model_name = "Qwen/Qwen3-4B-Thinking-2507"
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",  # dùng "cuda" nếu có GPU
            # torch_dtype="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,  # giảm nửa dung lượng
            low_cpu_mem_usage=True,  # load tối ưu RAM
        )

    def clean_html(self, raw_html: str) -> str:
        """Lọc bỏ HTML tag, script, style, và ký tự thừa."""
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text).strip()

    def json_extract(self, raw_text: str) -> str:
        match = re.search(r"json\s*(\{.*?\})\s*", raw_text, re.DOTALL)

        if not match:
            print("❌ Không tìm thấy JSON trong văn bản.")
        else:
            json_str = match.group(1)
        try:
            # ---- Bước 2: Parse JSON ----
            data = json.loads(json_str)

            # ---- Bước 3: Lọc các trường cần thiết ----
            filtered = {
                k: data[k]
                for k in ["is_related", "category", "reason"]
                if k in data
            }

            # ---- Bước 4: In kết quả đẹp ----
            res = json.dumps(filtered, indent=2, ensure_ascii=False)
            print(res)
            return res

        except json.JSONDecodeError as e:
            print("❌ JSON không hợp lệ:", e)

    def question(self, q: str):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a content classification model. "
                    "You must respond ONLY in strict JSON format, nothing else."
                )
            },
            {
                "role": "user",
                "content": f"""
                            Analyze the following text and determine if it is related to:
                            1. GAMBLING – includes betting, casinos, poker, slots, lotteries, sports betting, etc.
                            2. ADULT – includes sexual, pornographic, erotic, or 18+ dating content.
                            
                            Return ONLY a JSON in this format:
                            {{
                              "is_related": true or false,
                              "category": "GAMBLING" or "ADULT" or "NONE",
                              "reason": "Short explanation (in English)"
                            }}
                            
                            If both appear, choose the most dominant.
                            
                            Text (Vietnamese, but analyze meaning in English):
                            {q}
                            
                            Now output ONLY the JSON below, nothing else.
                            """
            }
        ]

        # ✅ Dùng chat template để Qwen hiểu đúng cấu trúc hội thoại
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0,
            eos_token_id=self.tokenizer.eos_token_id
        )

        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 🔎 Lọc phần JSON ra khỏi toàn bộ text (phòng khi model in thêm lời)
        # json_match = re.search(r"\{.*\}", output_text, flags=re.DOTALL)
        # if json_match:
        #     json_text = json_match.group(0)
        #     try:
        #         response = json.loads(json_text)
        #     except json.JSONDecodeError:
        #         response = {"is_related": None, "category": "NONE", "reason": json_text}
        # else:
        #     response = {"is_related": None, "category": "NONE", "reason": output_text}
        try:
            json_text = self.json_extract(output_text)
            response = json.loads(json_text)
        except json.JSONDecodeError:
            response = {"is_related": None, "category": "NONE", "reason": json_text}
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return response
