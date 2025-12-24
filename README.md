# ModelMix-Text-Generator

Python script that compares text generation across multiple open-source language models. Accepts user input, generates responses using Hugging Face models, tests three temperature values, and saves all outputs in structured JSON for easy comparison and analysis.

## Features

- ✅ Accepts user input via command line or interactive prompt
- ✅ Uses two open-source Hugging Face models (GPT-2 and DistilGPT-2)
- ✅ Tests 3 temperature values (0.3, 0.7, 1.0)
- ✅ Saves all outputs to JSON format
- ✅ Easy to extend with additional models

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode
```bash
python text_generator.py
```

### Command Line Mode
```bash
python text_generator.py --input "Your prompt here" --output results.json
```

### Arguments
- `--input`: User input prompt for text generation (optional, will prompt if not provided)
- `--output`: Output JSON file path (default: `generation_results.json`)

## Output Format

The script generates a JSON file with the following structure:
```json
{
  "user_input": "Your prompt",
  "timestamp": "2024-01-01T12:00:00",
  "models": {
    "gpt2": {
      "display_name": "GPT-2",
      "temperature_results": {
        "0.3": {
          "temperature": 0.3,
          "generated_text": "...",
          "length": 123
        },
        ...
      }
    },
    ...
  }
}
```

## Models Used

- **GPT-2**: OpenAI's GPT-2 model (124M parameters)
- **DistilGPT-2**: Distilled version of GPT-2 (82M parameters, faster inference)
