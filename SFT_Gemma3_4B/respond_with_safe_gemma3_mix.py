import os
import sys
import torch
torch.set_float32_matmul_precision('high')
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel

def load_finetuned_model(base_model_id = "google/gemma-3-4b-it", adapter_dir = "gemma3-4b-it_safety_mix_qlora",):
    if not torch.cuda.is_available():
        print("no cuda", file=sys.stderr)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_auth_token=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="cuda",
        use_auth_token=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        device_map="cuda",
    )
    
    model.eval()
    print("model loaded")
    return model, tokenizer

def get_llm_response(model, tokenizer, question):
    clean_question = question.strip().strip('"“”')
    prompt = f"Instruction: {clean_question}\n Answer:\n"
    # prompt = f"Istruzione: {clean_question}\n Risposta:\n"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=150,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            min_p = 0.0
        )

    resp_tokens = output_tokens[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(resp_tokens, skip_special_tokens=True).strip()

def process_questions(question_path, output_path, model, tokenizer, progress_file):
    try:
        if os.path.exists(output_path):
            df = pd.read_excel(output_path, engine="openpyxl")
            print(f"resume from {output_path}")
        else:
            df = pd.read_excel(question_path, engine="openpyxl")
            print(f"load {len(df)} questions from {question_path}")
            if "response" not in df.columns:
                df.insert(2, "response", "")

        start_idx = 0
        if os.path.exists(progress_file):
            try:
                start_idx = int(open(progress_file).read().strip()) or 0
                print(f"continue from row {start_idx}.")
            except Exception as e:
                print(f"progress file error {e}", file=sys.stderr)

        for i in tqdm(range(start_idx, len(df)), desc="generating responses", initial=start_idx, total=len(df)):
            question = str(df.iloc[i, 1])
            if question.strip():
                response = get_llm_response(model, tokenizer, question)
                if not response.strip():
                    print("no response")
                else:
                    df.iloc[i, 2] = response
            else:
                df.iloc[i, 2] = ""

            df.to_excel(output_path, index=False, engine='openpyxl')
            with open(progress_file, 'w') as f:
                f.write(str(i + 1))

        print(f"all responses saved at {output_path}")

    except FileNotFoundError:
        print(f"{question_path} not found", file=sys.stderr)
    except Exception as e:
        print(f"error {e}", file=sys.stderr)

if __name__ == "__main__":
    QUESTIONS_PATH = "dataset/harmful_question_clean_en_250_ts.xlsx"
    OUTPUT_PATH = "dataset/safe_gemma3_4b_mix_250_responses_en.xlsx"
    PROG_FILE = "safe_gemma3_4b_mix_progress_en.txt"
    # QUESTIONS_PATH = "dataset/harmful_question_clean_it_250_ts.xlsx"
    # OUTPUT_PATH = "dataset/safe_gemma3_4b_mix_250_responses_it.xlsx"
    # PROG_FILE = "safe_gemma3_4b_mix_progress_it.txt"
    
    model, tokenizer = load_finetuned_model()
    process_questions(
        question_path=QUESTIONS_PATH,
        output_path=OUTPUT_PATH,
        model=model,
        tokenizer=tokenizer,
        progress_file=PROG_FILE,
    )
