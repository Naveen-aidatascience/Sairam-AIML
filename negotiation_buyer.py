# tes.py

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import random

# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class Product:
    name: str
    category: str
    quantity: int
    quality_grade: str
    origin: str
    base_market_price: int
    attributes: Dict[str, Any]

@dataclass
class NegotiationContext:
    product: Product
    your_budget: int
    current_round: int
    seller_offers: List[int]
    your_offers: List[int]
    messages: List[Dict[str, str]]

class DealStatus(Enum):
    ONGOING = "ongoing"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    TIMEOUT = "timeout"

# ============================================
# BASE AGENT
# ============================================

class BaseBuyerAgent:
    def __init__(self, name: str):
        self.name = name
        self.personality = self.define_personality()
    def define_personality(self) -> Dict[str, Any]:
        pass
    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        pass
    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        pass
    def get_personality_prompt(self) -> str:
        pass

# ============================================
# ADVANCED BUYER AGENT
# ============================================

class AdvancedBuyerAgent(BaseBuyerAgent):
    def __init__(self, name: str, past_deals: list = None, alternatives: list = None):
        super().__init__(name)
        self.past_deals = past_deals or []
        self.alternatives = alternatives or []
        self.reset()

    def reset(self):
        self.round_num = 0
        self.current_offer = 0
        self.last_seller_price = 0
        self.reservation_price = 0
        self.target_price = 0

    def define_personality(self) -> Dict[str, Any]:
        return {
            "personality_type": "adaptive-pro",
            "traits": ["analytical", "patient", "profit-oriented", "strategic"],
            "negotiation_style": "Learns from past deals, predicts seller behavior, maximizes profit",
            "catchphrases": ["Let's find a fair deal.", "I aim to be reasonable.", "Can we meet halfway?"]
        }

    def generate_opening_offer(self, context: NegotiationContext) -> Tuple[int, str]:
        market_ref = context.product.base_market_price
        if self.past_deals:
            avg_past_price = sum(self.past_deals) / len(self.past_deals)
            opening_price = int(min(avg_past_price, market_ref * 0.75))
        else:
            opening_price = int(market_ref * 0.7)
        opening_price = min(opening_price, context.your_budget)
        self.current_offer = opening_price
        return opening_price, f"My opening offer for {context.product.name} is ₹{opening_price}. Let's find a fair deal."

    def respond_to_seller_offer(self, context: NegotiationContext, seller_price: int, seller_message: str) -> Tuple[DealStatus, int, str]:
        self.round_num += 1
        self.last_seller_price = seller_price

        if self.alternatives and seller_price > min(self.alternatives):
            return DealStatus.REJECTED, 0, "I have a better alternative available, cannot accept this price."

        market_price = context.product.base_market_price
        self.reservation_price = context.your_budget
        self.target_price = int(min(self.reservation_price * 0.85, market_price * 0.95))

        if seller_price <= self.target_price:
            self.past_deals.append(seller_price)
            return DealStatus.ACCEPTED, seller_price, f"Deal accepted at ₹{seller_price}! That works well for both of us."

        if self.round_num >= 10:
            if seller_price <= self.reservation_price:
                self.past_deals.append(seller_price)
                return DealStatus.ACCEPTED, seller_price, f"Last round acceptance at ₹{seller_price}."
            else:
                return DealStatus.REJECTED, 0, "Negotiation failed. Price too high."

        if len(context.seller_offers) >= 2:
            last = context.seller_offers[-1]
            prev = context.seller_offers[-2]
            trend = last - prev
            predicted_next = last - trend * 0.7
            next_offer = int(min(predicted_next, self.target_price,
                                 self.current_offer + (self.reservation_price - self.current_offer) // max(10 - self.round_num, 1)))
        else:
            next_offer = int(self.current_offer + (self.reservation_price - self.current_offer) // max(10 - self.round_num, 1))

        self.current_offer = min(next_offer, self.reservation_price, seller_price - 1)
        return DealStatus.ONGOING, self.current_offer, f"My counter-offer is ₹{self.current_offer}. Can we reach an agreement?"

# ============================================
# MOCK SELLER
# ============================================

class MockSellerAgent:
    def __init__(self, min_price: int):
        self.min_price = min_price
    def get_opening_price(self, product: Product) -> Tuple[int, str]:
        price = int(product.base_market_price * 1.5)
        return price, f"Seller asking ₹{price} for {product.name}."
    def respond_to_buyer(self, buyer_offer: int, round_num: int) -> Tuple[int, str, bool]:
        if buyer_offer >= self.min_price:
            return buyer_offer, f"You have a deal at ₹{buyer_offer}!", True
        counter = max(self.min_price, int(buyer_offer * 1.05))
        return counter, f"Seller counter-offer: ₹{counter}", False

# ============================================
# NEGOTIATION TEST
# ============================================

def run_negotiation_test(agent, product, budget, seller_min):
    seller = MockSellerAgent(seller_min)
    context = NegotiationContext(product, budget, 0, [], [], [])

    s_price, s_msg = seller.get_opening_price(product)
    context.seller_offers.append(s_price)
    context.messages.append({"role": "seller", "message": s_msg})

    deal_made = False
    final_price = 0
    rounds_taken = 0

    for round_num in range(10):
        context.current_round = round_num + 1
        if round_num == 0:
            b_offer, b_msg = agent.generate_opening_offer(context)
            status = DealStatus.ONGOING
        else:
            status, b_offer, b_msg = agent.respond_to_seller_offer(context, s_price, s_msg)

        context.your_offers.append(b_offer)
        context.messages.append({"role": "buyer", "message": b_msg})

        rounds_taken += 1

        if status == DealStatus.ACCEPTED:
            deal_made = True
            final_price = s_price
            break

        s_price, s_msg, s_accept = seller.respond_to_buyer(b_offer, round_num)
        context.seller_offers.append(s_price)
        context.messages.append({"role": "seller", "message": s_msg})

        if s_accept:
            deal_made = True
            final_price = b_offer
            break

    savings = budget - final_price if deal_made else 0
    below_market = (product.base_market_price - final_price) / product.base_market_price * 100 if deal_made else 0

    return {
        "deal_made": deal_made,
        "final_price": final_price,
        "savings": savings,
        "rounds": rounds_taken,
        "below_market": below_market
    }

# ============================================
# TEST RUN
# ============================================

def test_your_agent():
    test_products = [
        Product("Alphonso Mangoes", "Mangoes", 100, "A", "Ratnagiri", 180000, {}),
        Product("Kesar Mangoes", "Mangoes", 150, "B", "Gujarat", 150000, {})
    ]

    agent = AdvancedBuyerAgent("TestBuyer")

    print("="*60)
    print(f"TESTING YOUR AGENT: {agent.name}")
    print(f"Personality: {agent.personality['personality_type']}")
    print("="*60)

    total_savings = 0
    deals_made = 0

    for product in test_products:
        for scenario in ["easy", "medium", "hard"]:
            if scenario == "easy":
                budget = int(product.base_market_price * 1.2)
                seller_min = int(product.base_market_price * 0.8)
            elif scenario == "medium":
                budget = int(product.base_market_price * 1.0)
                seller_min = int(product.base_market_price * 0.85)
            else:
                budget = int(product.base_market_price * 0.9)
                seller_min = int(product.base_market_price * 0.82)

            result = run_negotiation_test(agent, product, budget, seller_min)
            if result["deal_made"]:
                deals_made += 1
                total_savings += result["savings"]
                print(f"\nTest: {product.name} - {scenario} scenario")
                print(f"Your Budget: ₹{budget:,} | Market Price: ₹{product.base_market_price:,}")
                print(f"✅ DEAL at ₹{result['final_price']:,} in {result['rounds']} rounds")
                print(f"   Savings: ₹{result['savings']:,} ({result['savings']/budget*100:.1f}%)")
                print(f"   Below Market: {result['below_market']:.1f}%")
            else:
                print(f"❌ {product.name}-{scenario}: No deal")

    print("="*60)
    print(f"SUMMARY")
    print(f"Deals Completed: {deals_made}/6")
    print(f"Total Savings: ₹{total_savings:,}")
    success_rate = deals_made / 6 * 100
    print(f"Success Rate: {success_rate:.1f}%")
    print("="*60)

if __name__ == "__main__":
    test_your_agent()
