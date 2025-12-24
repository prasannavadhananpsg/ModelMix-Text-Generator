"""
ModelMix Text Generator
Generates text using multiple open-source models with different temperature values.
"""

import json
import argparse
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import torch

# Model configurations
MODELS = {
    "gpt2": {
        "model_name": "gpt2",
        "display_name": "GPT-2"
    },
    "distilgpt2": {
        "model_name": "distilgpt2",
        "display_name": "DistilGPT-2"
    }
}

# Temperature values to test
TEMPERATURES = [0.3, 0.7, 1.0]

# Default generation parameters
MAX_LENGTH = 150
NUM_RETURN_SEQUENCES = 1


def load_model(model_key):
    """Load a Hugging Face model and tokenizer."""
    config = MODELS[model_key]
    print(f"Loading {config['display_name']}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        model = AutoModelForCausalLM.from_pretrained(config["model_name"])
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Set model to evaluation mode
        model.eval()
        
        print(f"✓ {config['display_name']} loaded successfully")
        return model, tokenizer, config["display_name"]
    except Exception as e:
        print(f"✗ Error loading {config['display_name']}: {str(e)}")
        raise


def generate_text(model, tokenizer, prompt, temperature, max_length=MAX_LENGTH):
    """Generate text using the model with specified temperature."""
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            num_return_sequences=NUM_RETURN_SEQUENCES,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            top_k=50
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the prompt)
    generated_part = generated_text[len(prompt):].strip()
    
    return generated_part


def run_generation(user_input):
    """Run text generation with all models and temperature values."""
    results = {
        "user_input": user_input,
        "timestamp": datetime.now().isoformat(),
        "models": {}
    }
    
    # Load all models
    loaded_models = {}
    for model_key in MODELS.keys():
        try:
            model, tokenizer, display_name = load_model(model_key)
            loaded_models[model_key] = {
                "model": model,
                "tokenizer": tokenizer,
                "display_name": display_name
            }
        except Exception as e:
            print(f"Failed to load {model_key}: {str(e)}")
            continue
    
    if not loaded_models:
        print("No models were loaded successfully. Exiting.")
        return None
    
    # Generate text for each model and temperature
    for model_key, model_data in loaded_models.items():
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        display_name = model_data["display_name"]
        
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
                generated_text = generate_text(model, tokenizer, user_input, temp)
                results["models"][model_key]["temperature_results"][str(temp)] = {
                    "temperature": temp,
                    "generated_text": generated_text,
                    "length": len(generated_text)
                }
                print(f"Generated: {generated_text[:100]}..." if len(generated_text) > 100 else f"Generated: {generated_text}")
            except Exception as e:
                print(f"Error generating with temperature {temp}: {str(e)}")
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
    
    # Get user input
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
    
    # Run generation
    results = run_generation(user_input)
    
    if results:
        # Save results
        save_to_json(results, args.output)
        
        # Print summary
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"Input: {results['user_input']}")
        print(f"Models tested: {len(results['models'])}")
        print(f"Temperature values: {', '.join(map(str, TEMPERATURES))}")
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

