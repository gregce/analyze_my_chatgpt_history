# ChatGPT History Analyzer

Analyze your ChatGPT conversation history with visualizations and statistics.

## Features

- Message frequency analysis (weekly and monthly)
- Conversation frequency tracking
- Model usage statistics
- Detailed message and conversation counts
- Role-based message analysis (user vs assistant)
- Interactive visualizations

## Getting Your ChatGPT Data

1. Go to ChatGPT (https://chat.openai.com)
2. Click the top right profile icon → Settings → Data Controls → Export Data
3. You'll receive a zip file containing `conversations.json`
4. Copy `conversations.json` into this project's directory

## Installation

This project uses Poetry for dependency management. Make sure you have Poetry installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then install the project dependencies:

```bash
poetry install
```

## Usage

Run the analysis:

```bash
poetry run python -m chatgpt_analysis conversations.json
```

The script will create a timestamped directory (e.g., `chat_analysis_20240315_143022`) containing:
- `analysis_results.txt`: Summary statistics
- `chat_frequency.png`: Weekly conversation frequency visualization
- `message_frequency.png`: Weekly message frequency visualization
- `monthly_messages.png`: Monthly message count visualization
- `model_usage.png`: Model usage over time
- `model_distribution.txt`: Detailed model usage statistics

## Requirements

- Python 3.8+
- Poetry 