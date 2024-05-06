# 30_days_ml_project
Project for 30 Days ML Challenge

## Data Generation

This project focuses on generating synthetic conversation data to train a classification model for the "Break Into Data" Discord server. The `gen_discord_conv.py` script is used to generate synthetic conversations based on a prompt describing the Discord server's purpose, channels, and typical conversation topics.

The script uses the LangChain library to interface with various language models (Groq, Anthropic, or Google Generative AI) and generate structured conversation data. Each conversation is represented as a list of messages, with each message containing a user name, message content, message type (question, answer, comment), and a unique message ID.

To generate synthetic conversations, run:
```
python gen_discord_conv.py
```
This will create a `conversations.csv` file with the generated conversation data.

## Data Cleaning

[Placeholder for data cleaning steps]

## Classification Models

[Placeholder for classification model details]

## Evaluation

[Placeholder for evaluation metrics and techniques]

## Presentation

[Placeholder for front-end application details]

## Dependencies

- Python 3.7+
- LangChain
- [List other dependencies]

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (e.g., API keys for language models)
4. Run the script: `python gen_discord_conv.py`