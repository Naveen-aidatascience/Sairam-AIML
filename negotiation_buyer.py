"""
===========================================
AI NEGOTIATION AGENT - INTERVIEW TEMPLATE

===========================================
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod
import random

# ============================================
# PART 1: DATA STRUCTURES (DO NOT MODIFY)
# ============================================

@dataclass
class Product:
    """Product being negotiated"""
    name: str
    category: str
    quantity: int
    quality_grade: str  # 'A', 'B', or 'Export'
    origin: str
    base_market_price: int  # Reference price for this product
    attributes: Dict[str, Any]

@dataclass
class NegotiationContext:
    """Current negotiation state"""
    product: Product
    your_budget: int  # Your maximum budget (NEVER exceed this)
    current_round: int
    seller_offers: List[int]  # History of seller's offers
    your_offers: List[int]  # History of your offers
    messages: List[Dict[str, str]]  # Full conversation history

class DealStatus(Enum):
    ONGOING = "ongoing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"

# ============================================
# PART 2: BASE AGENT CLASS (DO NOT MODIFY)
# ============================================

class BaseBuyerAgent(ABC):
    """Base class for all buyer agents"""
    def __init__(self, name: str):
        self.name = name
        self.personality = self.define_personality()

    @abstractmethod
    def define_personality(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        pass

    @abstractmethod
    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        pass

    @abstractmethod
    def get_personality_prompt(self) -> str:
        pass

# ============================================
# PART 3: LLM + MEMORY ADAPTERS (NEW)
# ============================================

# ---- Ollama adapter (Llama 3) ----
class OllamaMessenger:
    """
    Thin wrapper over ollama.chat for consistent prompts.
    LLM writes POLISHED MESSAGES ONLY; we keep numbers and decisions purely in Python.
    """
    def __init__(self, model: str = "llama3", temperature: float = 0.65):
        try:
            import ollama  # local server-backed
        except Exception as e:
            raise RuntimeError(
                "Could not import 'ollama'. Install with `pip install ollama` "
                "and ensure `ollama serve` is running."
            ) from e
        self._ollama = __import__("ollama")
        self.model = model
        self.temperature = temperature

    def craft_message(self, role_goal: str, personality: Dict[str, Any], context: Dict[str, Any], speak_for: str = "buyer") -> str:
        system = (
            "You are an expert business negotiator acting as the BUYER. "
            "Write concise, professional, polite messages (1–2 sentences). "
            "NEVER invent numbers; ONLY reference prices I give you. "
            "Stay consistent with the provided personality and catchphrases."
        )
        # Keep payload compact but informative
        user = (
            f"ROLE GOAL: {role_goal}\n"
            f"PERSONALITY:\n{json.dumps(personality, ensure_ascii=False)}\n\n"
            f"CONTEXT:\n{json.dumps(context, ensure_ascii=False)}\n\n"
            f"Write one short {speak_for} message. No markdown, no emojis."
        )
        resp = self._ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={"temperature": self.temperature},
        )
        return resp["message"]["content"].strip()

# ---- Concordia memory (optional) ----
class DealMemory:
    """
    Lightweight semantic memory for seller behaviors, backed by Concordia if available.
    Falls back to simple string store if gdm-concordia (and its deps) aren't present.
    """
    def __init__(self):
        self.enabled = False
        self.events: List[str] = []
        try:
            # Minimal import: we won't spin up full social sims, just use storage if present.
            import importlib
            self._cd = importlib.import_module("concordia")
            # Some Concordia distributions expose memory stores differently.
            # We keep a tiny fallback vector store using bag-of-words if unavailable.
            self.enabled = True
            self.store = []  # (text, bow) pairs
        except Exception:
            self.enabled = False
            self.store = []

    def _to_bow(self, text: str) -> Dict[str, int]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        bow: Dict[str, int] = {}
        for w in words:
            bow[w] = bow.get(w, 0) + 1
        return bow

    def _sim(self, a: Dict[str, int], b: Dict[str, int]) -> float:
        # Cosine on BOW
        import math
        if not a or not b:
            return 0.0
        dot = sum(a.get(k,0)*b.get(k,0) for k in set(a)|set(b))
        na = math.sqrt(sum(v*v for v in a.values()))
        nb = math.sqrt(sum(v*v for v in b.values()))
        return dot/(na*nb) if na and nb else 0.0

    def add_event(self, text: str):
        self.events.append(text)
        self.store.append((text, self._to_bow(text)))

    def recall(self, query: str, k: int = 3) -> List[str]:
        qbow = self._to_bow(query)
        scored = [(self._sim(qbow, bow), txt) for (txt, bow) in self.store]
        scored.sort(reverse=True, key=lambda t: t[0])
        return [t[1] for t in scored[:k] if t[0] > 0.05]

# ============================================
# PART 4: YOUR IMPLEMENTATION (UPGRADED)
# ============================================

class YourBuyerAgent(BaseBuyerAgent):
    """
    LLM-backed Buyer Agent:
    - Personality-driven messaging via Llama 3 (Ollama)
    - Deterministic strategy for offers/decisions
    - Optional Concordia-like memory for seller behavior recall
    """

    def __init__(self, name: str, model: str = "llama3", temperature: float = 0.65):
        super().__init__(name)
        self.messenger = OllamaMessenger(model=model, temperature=temperature)
        self.memory = DealMemory()
        # Tunables
        self.max_rounds = 10
        self.min_discount_vs_market = 0.05  # target >=5% below market when possible
        self.target_discount_vs_market = 0.10  # aim ~10% below market if feasible

    # -------- Personality --------
    def define_personality(self) -> Dict[str, Any]:
        return {
            "personality_type": "adaptive-analytical",
            "traits": ["data-driven", "calm", "patient", "collaborative", "value-focused"],
            "negotiation_style": (
                "Anchors reasonably below market, concedes in small steps, "
                "rewards reciprocity, and closes quickly when targets are met."
            ),
            "catchphrases": [
                "Let's land on a fair number.",
                "I’m focused on value, not just price.",
                "Happy to close if we’re both comfortable."
            ]
        }

    def get_personality_prompt(self) -> str:
        return (
            "I am an adaptive, analytical buyer. I speak politely and concisely, "
            "justify numbers with market references, and suggest collaborative middle grounds. "
            "I never bluff; I reward meaningful concessions and propose clear next steps. "
            "Phrases I use: 'Let's land on a fair number.', 'I’m focused on value, not just price.', "
            "'Happy to close if we’re both comfortable.'"
        )

    # -------- Strategy helpers (deterministic) --------
    def calculate_fair_price(self, product: Product, budget: int) -> Tuple[int, int]:
        """
        Returns (target_price, reservation_price)
        - target_price: ideal closing price if seller is cooperative
        - reservation_price: hard cap (<= budget)
        """
        market = product.base_market_price
        reservation_price = int(min(budget, market * (1.0 - self.min_discount_vs_market)))
        target_price = int(min(reservation_price, market * (1.0 - self.target_discount_vs_market)))
        # Ensure sane ordering
        target_price = min(target_price, reservation_price)
        return target_price, reservation_price

    def _message(self, role_goal: str, context: NegotiationContext, extras: Dict[str, Any]) -> str:
        convo_tail = context.messages[-6:]  # keep short for context
        mem = []
        if self.memory.events:
            # Query with current seller last move
            q = extras.get("seller_message") or json.dumps(convo_tail[-1] if convo_tail else {})
            mem = self.memory.recall(f"seller behavior: {q}", k=3)
        ctx = {
            "product": asdict(context.product),
            "your_budget": context.your_budget,
            "current_round": context.current_round,
            "seller_offers": context.seller_offers[-5:],
            "your_offers": context.your_offers[-5:],
            "recent_messages": convo_tail,
            "memory": mem,
            **extras,
        }
        return self.messenger.craft_message(role_goal=role_goal, personality=self.personality, context=ctx)

    # -------- Opening --------
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        market = context.product.base_market_price
        target, reservation = self.calculate_fair_price(context.product, context.your_budget)

        # Adaptive opening: 70–75% of market, never exceed budget
        grade_factor = {"Export": 0.8, "A": 0.72, "B": 0.68}.get(context.product.quality_grade, 0.7)
        opening = int(market * grade_factor)
        opening = min(opening, context.your_budget)

        # Store memory cue
        self.memory.add_event(
            f"Opening anchored at {opening} for {context.product.name} "
            f"(market={market}, grade={context.product.quality_grade})"
        )

        msg = self._message(
            role_goal="Propose an opening offer anchored below market; be collaborative and confident.",
            context=context,
            extras={"opening_price": opening, "market_price": market, "target_price": target, "reservation_price": reservation}
        )
        return opening, msg

    # -------- Response loop --------
    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        current_round = context.current_round
        market = context.product.base_market_price
        target, reservation = self.calculate_fair_price(context.product, context.your_budget)

        # Log seller behavior for memory
        self.memory.add_event(f"Round {current_round}: Seller offered {seller_price}; said: {seller_message}")

        # If seller gives us target or better -> accept
        if seller_price <= target and seller_price <= context.your_budget:
            msg = self._message(
                role_goal="Accept warmly and close; emphasize mutual value.",
                context=context,
                extras={"seller_price": seller_price, "decision": "accept", "reason": "meets target"}
            )
            return DealStatus.ACCEPTED, seller_price, msg

        # Near timeout? Consider accepting up to reservation/budget if value is decent
        if current_round >= self.max_rounds - 1:
            if seller_price <= reservation:
                msg = self._message(
                    role_goal="Accept due to time constraints; keep it gracious.",
                    context=context,
                    extras={"seller_price": seller_price, "decision": "accept", "reason": "endgame within reservation"}
                )
                return DealStatus.ACCEPTED, seller_price, msg
            else:
                # Final firm counter just under reservation or under seller's number
                counter = min(max(context.your_offers[-1] if context.your_offers else 0, int(reservation * 0.98)),
                              seller_price - 1, reservation)
                msg = self._message(
                    role_goal="Make a final concise counter-offer; communicate clear walk-away.",
                    context=context,
                    extras={"seller_price": seller_price, "your_offer": counter, "decision": "final-counter"}
                )
                return DealStatus.ONGOING, counter, msg

        # Compute suggested counter path (concession curve)
        last_own = context.your_offers[-1] if context.your_offers else int(market * 0.7)
        # If seller concedes, we concede a bit; otherwise hold tighter
        seller_trend = 0
        if len(context.seller_offers) >= 2:
            seller_trend = context.seller_offers[-1] - context.seller_offers[-2]

        # Base step: fraction of gap to reservation, decreasing each round
        rounds_left = max(1, self.max_rounds - current_round)
        step = max(1, (reservation - last_own) // (rounds_left + 2))
        if seller_trend < 0:  # seller moved toward us
            step = int(step * 1.15)
        else:
            step = int(step * 0.9)

        counter = min(last_own + step, reservation, seller_price - 1)

        # Guardrails: never go above budget/reservation
        counter = min(counter, context.your_budget, reservation)
        counter = max(counter, min(last_own, reservation))  # avoid backwards jumps

        # If seller offer is barely above our target, try mid-point to close faster
        if seller_price < int(target * 1.05) and counter < seller_price:
            midpoint = (seller_price + counter) // 2
            counter = min(midpoint, reservation)

        # Compose message
        msg = self._message(
            role_goal="Counter with rationale tied to market and quality; invite movement to close gap.",
            context=context,
            extras={
                "seller_price": seller_price,
                "seller_message": seller_message,
                "your_offer": counter,
                "market_price": market,
                "target_price": target,
                "reservation_price": reservation
            }
        )
        return DealStatus.ONGOING, counter, msg

# ============================================
# PART 5: SIMPLE REFERENCE AGENT (UNCHANGED)
# ============================================

class ExampleSimpleAgent(BaseBuyerAgent):
    def define_personality(self) -> Dict[str, Any]:
        return {
            "personality_type": "cautious",
            "traits": ["careful", "budget-conscious", "polite"],
            "negotiation_style": "Makes small incremental offers, very careful with money",
            "catchphrases": ["Let me think about that...", "That's a bit steep for me"]
        }

    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        opening = int(context.product.base_market_price * 0.6)
        opening = min(opening, context.your_budget)
        return opening, f"I'm interested, but ₹{opening} is what I can offer. Let me think about that..."

    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        if seller_price <= context.your_budget and seller_price <= context.product.base_market_price * 0.85:
            return DealStatus.ACCEPTED, seller_price, f"Alright, ₹{seller_price} works for me!"
        last_offer = context.your_offers[-1] if context.your_offers else 0
        counter = min(int(last_offer * 1.1), context.your_budget)
        if counter >= seller_price * 0.95:
            counter = min(seller_price - 1000, context.your_budget)
            return DealStatus.ONGOING, counter, f"That's a bit steep for me. How about ₹{counter}?"
        return DealStatus.ONGOING, counter, f"I can go up to ₹{counter}, but that's pushing my budget."

    def get_personality_prompt(self) -> str:
        return """
        I am a cautious buyer who is very careful with money. I speak politely but firmly.
        I often say things like 'Let me think about that' or 'That's a bit steep for me'.
        I make small incremental offers and show concern about my budget.
        """

# ============================================
# PART 6: SELLER + TEST HARNESS (UNCHANGED)
# ============================================

class MockSellerAgent:
    """A simple mock seller for testing your agent"""
    def __init__(self, min_price: int, personality: str = "standard"):
        self.min_price = min_price
        self.personality = personality

    def get_opening_price(self, product: Product) -> Tuple[int, str]:
        price = int(product.base_market_price * 1.5)
        return price, f"These are premium {product.quality_grade} grade {product.name}. I'm asking ₹{price}."

    def respond_to_buyer(self, buyer_offer: int, round_num: int) -> Tuple[int, str, bool]:
        if buyer_offer >= self.min_price * 1.1:  # Good profit
            return buyer_offer, f"You have a deal at ₹{buyer_offer}!", True
        if round_num >= 8:  # Close to timeout
            counter = max(self.min_price, int(buyer_offer * 1.05))
            return counter, f"Final offer: ₹{counter}. Take it or leave it.", False
        else:
            counter = max(self.min_price, int(buyer_offer * 1.15))
            return counter, f"I can come down to ₹{counter}.", False

def run_negotiation_test(buyer_agent: BaseBuyerAgent, product: Product, buyer_budget: int, seller_min: int) -> Dict[str, Any]:
    seller = MockSellerAgent(seller_min)
    context = NegotiationContext(
        product=product, your_budget=buyer_budget,
        current_round=0, seller_offers=[], your_offers=[], messages=[]
    )

    # Seller opens
    seller_price, seller_msg = seller.get_opening_price(product)
    context.seller_offers.append(seller_price)
    context.messages.append({"role": "seller", "message": seller_msg})

    # Run negotiation
    deal_made = False
    final_price = None

    for round_num in range(10):  # Max 10 rounds
        context.current_round = round_num + 1

        # Buyer responds
        if round_num == 0:
            buyer_offer, buyer_msg = buyer_agent.generate_opening_offer(context)
            status = DealStatus.ONGOING
        else:
            status, buyer_offer, buyer_msg = buyer_agent.respond_to_seller_offer(context, seller_price, seller_msg)

        context.your_offers.append(buyer_offer)
        context.messages.append({"role": "buyer", "message": buyer_msg})

        if status == DealStatus.ACCEPTED:
            deal_made = True
            final_price = seller_price
            break

        # Seller responds
        seller_price, seller_msg, seller_accepts = seller.respond_to_buyer(buyer_offer, round_num)
        if seller_accepts:
            deal_made = True
            final_price = buyer_offer
            context.messages.append({"role": "seller", "message": seller_msg})
            break

        context.seller_offers.append(seller_price)
        context.messages.append({"role": "seller", "message": seller_msg})

    result = {
        "deal_made": deal_made,
        "final_price": final_price,
        "rounds": context.current_round,
        "savings": buyer_budget - final_price if deal_made else 0,
        "savings_pct": ((buyer_budget - final_price) / buyer_budget * 100) if deal_made else 0,
        "below_market_pct": ((product.base_market_price - final_price) / product.base_market_price * 100) if deal_made else 0,
        "conversation": context.messages
    }
    return result

# ============================================
# PART 7: TEST YOUR AGENT (UPDATED TO USE LLM AGENT)
# ============================================

def test_your_agent():
    test_products = [
        Product(
            name="Alphonso Mangoes", category="Mangoes",
            quantity=100, quality_grade="A", origin="Ratnagiri",
            base_market_price=180000,
            attributes={"ripeness": "optimal", "export_grade": True}
        ),
        Product(
            name="Kesar Mangoes", category="Mangoes",
            quantity=150, quality_grade="B", origin="Gujarat",
            base_market_price=150000,
            attributes={"ripeness": "semi-ripe", "export_grade": False}
        )
    ]

    # Use Llama3 (Ollama) with polite, strategic messaging
    your_agent = YourBuyerAgent("TestBuyer", model="llama3", temperature=0.65)

    print("="*60)
    print(f"TESTING YOUR AGENT: {your_agent.name}")
    print(f"Personality: {your_agent.personality['personality_type']}")
    print("="*60)

    total_savings = 0
    deals_made = 0

    for product in test_products:
        for scenario in ["easy", "medium", "hard"]:
            if scenario == "easy":
                buyer_budget = int(product.base_market_price * 1.2)
                seller_min = int(product.base_market_price * 0.8)
            elif scenario == "medium":
                buyer_budget = int(product.base_market_price * 1.0)
                seller_min = int(product.base_market_price * 0.85)
            else:  # hard
                buyer_budget = int(product.base_market_price * 0.9)
                seller_min = int(product.base_market_price * 0.82)

            print(f"\nTest: {product.name} - {scenario} scenario")
            print(f"Your Budget: ₹{buyer_budget:,} | Market Price: ₹{product.base_market_price:,}")

            result = run_negotiation_test(your_agent, product, buyer_budget, seller_min)

            if result["deal_made"]:
                deals_made += 1
                total_savings += result["savings"]
                print(f"✅ DEAL at ₹{result['final_price']:,} in {result['rounds']} rounds")
                print(f"   Savings: ₹{result['savings']:,} ({result['savings_pct']:.1f}%)")
                print(f"   Below Market: {result['below_market_pct']:.1f}%")
            else:
                print(f"❌ NO DEAL after {result['rounds']} rounds")

    print("\n" + "="*60)
    print("SUMMARY")
    print(f"Deals Completed: {deals_made}/6")
    print(f"Total Savings: ₹{total_savings:,}")
    print(f"Success Rate: {deals_made/6*100:.1f}%")
    print("="*60)

# ============================================
# PART 8: MAIN
# ============================================

if __name__ == "__main__":
    test_your_agent()
