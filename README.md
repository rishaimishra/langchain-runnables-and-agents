# LangChain Runnables, Tool Calling & Agents

Part of my **Founder's AI Learning Path** — learning AI engineering in public while running [SRV Technology](https://srvtechnology.com).

Chains are easy once you see Runnables as composable pipes. This repo works up from the basic building blocks (sequence, parallel, passthrough) to a full agent that calls tools and asks for human approval before acting — inspired by working through [AkarshVyas/Generative-AI-video3-](https://github.com/AkarshVyas/Generative-AI-video3-), rewritten to understand each layer rather than copy it.

## Building blocks

- `sequence_runnable.py` — the basic `prompt | model | parser` pipe
- `parallel_runnable.py` — `RunnableParallel` to run two independent chains at once and collect both results
- `passthrough_runnable.py` — `RunnablePassthrough` to carry the original input alongside a transformed branch (e.g. "generate code" + "explain that same code" in one chain)
- `custom_tool.py` — defining a tool from a plain Python function with the `@tool` decorator
- `tool_calling.py` — binding a tool to a model and manually handling the tool-call round trip
- `agent_with_approval.py` — a `create_agent` agent with multiple tools (weather + news lookup) and a `human_approval` middleware that pauses before every tool call

## Why the human-approval middleware is the interesting part

Giving an agent tools is the easy part. The part that actually matters for shipping anything real is controlling *when* it's allowed to act — `agent_with_approval.py` wraps every tool call in a confirmation step, so the agent proposes an action and a human decides whether it executes. That's the difference between a demo and something you'd trust unattended.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in the keys for whichever script you're running
python agent_with_approval.py
```

## Learning path

See the write-ups on [LinkedIn](https://www.linkedin.com/in/rishav-kumar-4441b3112/) for the reasoning behind each piece.
