from transformers import pipeline


class ZeroShot:

    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device="cpu"
        )

    def question(self, prompt):
        result = self.classifier(
            "vu88.net",
            candidate_labels=["GAMBLING", "ADULT", "NORMAL"]
        )
        # 👉 Lấy nhãn có xác suất cao nhất
        label = result["labels"][0]
        score = result["scores"][0]
        print(f"Predicted: {label} ({score:.2%})")
