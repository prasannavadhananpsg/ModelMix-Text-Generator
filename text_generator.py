"""
ModelMix Text Generator
Generates text using multiple open-source models with different temperature values.
"""

import json
import argparse
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import torch

MODELS = {
    "gpt2": {
        "model_name": "gpt2",
        "display_name": "GPT-2",
        "use_instruction_format": True
    },
    "distilgpt2": {
        "model_name": "distilgpt2",
        "display_name": "DistilGPT-2",
        "use_instruction_format": True
    }
}

TEMPERATURES = [0.3, 0.7, 1.0]

MAX_NEW_TOKENS = 200
NUM_RETURN_SEQUENCES = 1
REPETITION_PENALTY = 1.2


def load_model(model_key):
    """Load a Hugging Face model and tokenizer."""
    config = MODELS[model_key]
    print(f"Loading {config['display_name']}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        model = AutoModelForCausalLM.from_pretrained(config["model_name"])
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        
        print(f"✓ {config['display_name']} loaded successfully")
        return model, tokenizer, config["display_name"]
    except Exception as e:
        print(f"✗ Error loading {config['display_name']}: {str(e)}")
        raise


def format_prompt(user_input, use_instruction_format=True):
    """Format the user input as a better prompt for text generation.
    
    Note: The prompt is reformatted because base language models (like GPT-2) work better
    with natural text continuations rather than direct questions. This transformation helps
    achieve more coherent and relevant outputs from instruction-agnostic models.
    """
    if use_instruction_format:
        user_lower = user_input.lower().strip()
        
        topic = user_input
        for phrase in ["tell me", "give me", "what are", "what is", "explain", "describe"]:
            if phrase in user_lower:
                topic = user_input.lower().replace(phrase, "").strip()
                topic = topic.replace("?", "").strip()
                break
        
        if "scientific facts" in user_lower or "science" in user_lower:
            formatted = f"Scientific facts are fascinating. For example, "
        elif topic:
            formatted = f"Here are some important facts about {topic}. "
        else:
            formatted = f"Here are some facts. "
    else:
        formatted = user_input
    return formatted


def generate_text(model, tokenizer, prompt, temperature, max_new_tokens=MAX_NEW_TOKENS, use_instruction_format=True):
    """Generate text using the model with specified temperature."""
    formatted_prompt = format_prompt(prompt, use_instruction_format)
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0.1 else False,
            num_return_sequences=NUM_RETURN_SEQUENCES,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=REPETITION_PENALTY,
            top_p=0.95 if temperature > 0.5 else 0.9,
            top_k=50,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if formatted_prompt in generated_text:
        generated_part = generated_text.split(formatted_prompt, 1)[-1].strip()
    else:
        if prompt in generated_text:
            generated_part = generated_text.split(prompt, 1)[-1].strip()
        else:
            generated_part = generated_text.strip()
    
    if generated_part and not generated_part.endswith(('.', '!', '?')):
        last_period = generated_part.rfind('.')
        last_exclamation = generated_part.rfind('!')
        last_question = generated_part.rfind('?')
        last_sentence_end = max(last_period, last_exclamation, last_question)
        if last_sentence_end > len(generated_part) * 0.5:
            generated_part = generated_part[:last_sentence_end + 1]
    
    return generated_part


def run_generation(user_input):
    """Run text generation with all models and temperature values."""
    results = {
        "user_input": user_input,
        "timestamp": datetime.now().isoformat(),
        "models": {}
    }
    
    loaded_models = {}
    for model_key in MODELS.keys():
        try:
            model, tokenizer, display_name = load_model(model_key)
            loaded_models[model_key] = {
                "model": model,
                "tokenizer": tokenizer,
                "display_name": display_name,
                "use_instruction_format": MODELS[model_key].get("use_instruction_format", True)
            }
        except Exception as e:
            print(f"Failed to load {model_key}: {str(e)}")
            continue
    
    if not loaded_models:
        print("No models were loaded successfully. Exiting.")
        return None
    
    for model_key, model_data in loaded_models.items():
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        display_name = model_data["display_name"]
        use_instruction_format = model_data["use_instruction_format"]
        
        print(f"\n{'='*60}")
        print(f"Generating with {display_name}")
        print(f"{'='*60}")
        
        results["models"][model_key] = {
            "display_name": display_name,
            "temperature_results": {}
        }
        
        for temp in TEMPERATURES:
            print(f"\nTemperature: {temp}")
            print("-" * 60)
            
            try:
                generated_text = generate_text(
                    model, 
                    tokenizer, 
                    user_input, 
                    temp,
                    use_instruction_format=use_instruction_format
                )
                results["models"][model_key]["temperature_results"][str(temp)] = {
                    "temperature": temp,
                    "generated_text": generated_text,
                    "length": len(generated_text)
                }
                if generated_text:
                    display_text = generated_text[:300] + "..." if len(generated_text) > 300 else generated_text
                    print(f"Generated ({len(generated_text)} chars):\n{display_text}")
                else:
                    print("Generated: (empty response)")
            except Exception as e:
                print(f"Error generating with temperature {temp}: {str(e)}")
                import traceback
                traceback.print_exc()
                results["models"][model_key]["temperature_results"][str(temp)] = {
                    "temperature": temp,
                    "error": str(e)
                }
    
    return results


def save_to_json(results, output_file="generation_results.json"):
    """Save generation results to JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {output_file}")


def main():
    """Main function to handle user input and run generation."""
    parser = argparse.ArgumentParser(
        description="Generate text using multiple open-source models with different temperature values"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="User input prompt for text generation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generation_results.json",
        help="Output JSON file path (default: generation_results.json)"
    )
    
    args = parser.parse_args()
    
    if args.input:
        user_input = args.input
    else:
        print("Enter your prompt for text generation:")
        user_input = input("> ").strip()
    
    if not user_input:
        print("Error: No input provided. Exiting.")
        return
    
    print(f"\n{'='*60}")
    print(f"Input: {user_input}")
    print(f"{'='*60}\n")
    
    results = run_generation(user_input)
    
    if results:
        save_to_json(results, args.output)
        
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"Input: {results['user_input']}")
        print(f"Models tested: {len(results['models'])}")
        print(f"Temperature values: {', '.join(map(str, TEMPERATURES))}")
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

