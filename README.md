AI Negotiation Agent (Buyer-Side)

This project implements an AI-powered buyer negotiation agent that leverages Llama3 (via Ollama) for natural language messaging and an optional Concordia-inspired memory system for seller behavior recall.
The agent can negotiate with sellers using a structured, personality-driven strategy while respecting budgets and market prices.

Features

Adaptive Analytical Personality – calm, data-driven, value-focused buyer behavior.

Deterministic Offer Strategy – combines market references, concessions, and reservation pricing.

LLM Messaging (Ollama Llama3) – generates short, polished, professional buyer messages.

Memory Module (optional) – recalls past seller behaviors to adapt counteroffers.

Test Harness – mock seller with different difficulty levels (easy, medium, hard) for benchmarking.

Installation

Clone this repo.

git clone https://github.com/yourname/ai-negotiation-agent.git
cd ai-negotiation-agent


Install Python dependencies.

pip install -r requirements.txt


Requirements:

ollama (for Llama3 messaging)

dataclasses (if Python <3.7)

typing (standard for 3.9+)

Optional: concordia

Ensure Ollama server is running.

ollama serve
ollama pull llama3


Run the negotiation tests.

python negotiation_agent.py

Negotiation Algorithm

The buyer agent follows a deterministic + LLM hybrid strategy:

1. Initialization

Calculate fair target price (≈ 10% below market).

Calculate reservation price (≈ 5% below market or budget).

Generate opening offer based on quality grade and anchored at 68–80% of market price.

2. Offer Response Loop

Each round, the buyer decides between:

Accepting (if seller ≤ target or within reservation).

Counter-offering (gradual concessions).

Final Counter (firm last offer near reservation if nearing timeout).

3. Concession Curve

Buyer makes small concessions proportional to rounds left.

Rewards seller concessions with slightly larger steps.

If seller is near target, proposes a midpoint compromise.

4. Endgame

If timeout approaches:

Accepts if seller ≤ reservation.

Otherwise, makes a final firm counter and walks away if rejected.

 Strategy Summary

Anchoring: Opens below market but within reasonable bounds.

Reciprocity: Concedes more if seller shows movement.

Value Focus: References market, quality, and budget justification.

Time Pressure: Tightens offers near the final round.

Personality Consistency: Uses polite, concise, and professional phrasing with catchphrases:

"Let's land on a fair number."

"I’m focused on value, not just price."

"Happy to close if we’re both comfortable."

Test Scenarios

The test harness runs against mock sellers under 3 conditions:

Easy: High buyer budget, seller flexible.

Medium: Balanced budget vs. seller minimum.

Hard: Tight budget, aggressive seller.

