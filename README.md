# Confucius (OSS slice)

This repository contains a minimal, open-sourceable subset of Confucius. It is intended to be mirrored from the internal repository to GitHub via ShipIt.

## Quickstart: run the CLI

1) Create a conda environment and install dependencies

- From the repo root (this directory contains `confucius/` and `requirements.txt`):
  - Create and activate an environment
    - `conda create -n confucius python=3.12 -y`
    - `conda activate confucius`
  - Install Python dependencies
    - `pip install -r requirements.txt`

2) Configure provider credentials (choose one)

Confucius can talk to multiple LLM providers. Set the env vars for the provider you intend to use. The simplest is OpenAI:

- OpenAI (recommended for quick start):
  - `export OPENAI_API_KEY="<your_openai_key>"`

- Google Generative AI (via google-genai):
  - `export GOOGLE_API_KEY="<your_google_api_key>"`

- AWS Bedrock (via boto3): ensure your AWS credentials/region are configured
  - `export AWS_REGION=us-east-1`
  - and either `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` or a named profile.

- Azure OpenAI (if you configure Azure in code):
  - `export AZURE_OPENAI_API_KEY="<your_azure_key>"`
  - `export AZURE_OPENAI_ENDPOINT="https://<your-resource-name>.openai.azure.com"`

3) Run the CLI

- Basic usage (Python module):
  - `python -m confucius.cli.main code`

This launches a minimal REPL for the Code assistant. Type your prompt and press Enter to interact. Use Ctrl-C to interrupt a running generation; a second Ctrl-C exits.

## PEX build
Run the following to package a CF entrypoing into PEX binary, here -m can be any entrypoint of choice, -o is the output binary path
```
pex . \
  -r requirements.txt \
  -m scripts.run_cca \
  -o ~/workspace/sbv_test/app.pex \
  --python-shebang="/usr/bin/env python3"
```

## Layout

- confucius/
  - Package root for libraries and CLI scaffolding
- OSS docs
  - LICENSE, README.md, CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md

## SWE-Bench Pro batch evaluation

To run Confucius on SWE-Bench Pro problems at scale, use the batch evaluation script:

```bash
python scripts/run_batch_swebench.py [options]
```

See **[SWEBENCH_README.md](SWEBENCH_README.md)** for full documentation, including
prerequisites, parameter reference, example commands, results structure, and
troubleshooting.

## License

MIT — see LICENSE.
