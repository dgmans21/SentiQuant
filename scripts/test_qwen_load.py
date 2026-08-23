from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/Qwen3-4B",
    max_seq_length=2048,
    load_in_4bit=True,
)

print("MODEL LOAD OK")
print("device:", next(model.parameters()).device)
