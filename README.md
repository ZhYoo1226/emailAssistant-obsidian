# webinar-email-assistant-crewai-obsidian

This repository contains materials from the hands-on webinar "[Building Intelligent Agentic RAG with CrewAI and 
Qdrant](https://www.youtube.com/watch?v=soGB3UowTZ0)". It implements an agentic RAG system that uses your existing 
[Obsidian](https://obsidian.md/) Vault as a knowledge base to draft emails in Gmail Inbox.

## Software Stack

The project is built using Python and integrates with external services through their APIs. **The agentic behaviors are 
implemented using [CrewAI](https://www.crewai.com/), with [Qdrant](https://qdrant.tech/) serving as the memory layer for 
the system.**

## Prerequisites

You'll need access to a Qdrant instance, which can be set up in one of two ways:

1. Install and run it locally on your machine.
2. Sign up for a free account on [Qdrant Cloud](https://cloud.qdrant.io/).

Either option will provide you with a URL to connect to your instance. The cloud version will also provide an API key 
for authentication.

You'll need Python 3.10–3.12 installed, and we recommend using Poetry for dependency management. No GPU is required — 
LLM inference runs over an OpenAI-compatible gateway API, and embeddings run locally via fastembed (ONNX on CPU). Install 
all necessary libraries with a single command:

```shell
poetry install
```

The project runs all of its LLM tasks through a single OpenAI-compatible gateway serving DeepSeek models, and uses a 
local [fastembed](https://github.com/qdrant/fastembed) model for embeddings — so there are no separate LLM API keys to 
manage. You only need the gateway's base URL and API key (see Configuration below). CrewAI makes model switching 
straightforward, so you can point it at any other OpenAI-compatible endpoint if you prefer.

Optionally, you can use [AgentOps](https://agentops.ai/) for observability. This requires signing up for an account and 
obtaining an API key.

### Configuration

Create a `.env` file in the root directory with the following entries:

```dotenv
# LLM gateway (OpenAI-compatible)
GATEWAY_BASE_URL=https://api.upmore.net/v1
GATEWAY_API_KEY=YOUR_API_KEY
GATEWAY_MAIN_MODEL=deepseek-v4-pro
GATEWAY_FAST_MODEL=deepseek-v4-flash

# Local embedding (fastembed)
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5

# Qdrant configuration
QDRANT_LOCATION=http://localhost:6333
QDRANT_API_KEY=

# Observability with AgentOps
# Leave empty if you don't want to use AgentOps
AGENTOPS_API_KEY=

# Obsidian configuration
OBSIDIAN_VAULT_PATH=/path/to/vault
```

You can rename `.env.example` to `.env` and fill in the values, or set these as environment variables.

#### Google Cloud Credentials

Beyond environment variables, you'll need to set up Google Cloud Console credentials to access your Gmail account. 
Follow the [official documentation](https://developers.google.com/gmail/api/quickstart/python) to create a project and 
generate credentials. Download the credentials file and save it as `credentials.json` in the project's root directory.

#### Running Qdrant

To run Qdrant locally, use Docker. The following command pulls the latest Qdrant version and runs it:

```shell
bash scripts/run-qdrant.sh
```

Access the dashboard at http://localhost:6333/dashboard. Skip this step if you're using Qdrant Cloud.

## Usage

Run the application using:

```shell
poetry run python main.py
```

This launches two threads:

1. A filesystem watcher that monitors your Obsidian Vault for changes and updates the Qdrant index accordingly
2. A CrewAI agent that monitors your Gmail inbox and drafts responses based on knowledge stored in the Obsidian Vault

For a detailed explanation of the system and its components, refer to the [webinar 
recording](https://www.youtube.com/watch?v=soGB3UowTZ0) and the source code in this repository.
