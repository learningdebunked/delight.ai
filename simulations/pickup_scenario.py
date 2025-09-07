"""
Delight.AI - Real-world Pickup Simulation

This module simulates customer pickup scenarios to demonstrate the emotional intelligence
and cultural adaptation capabilities of the SEDS framework.
"""

import random
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

class OrderType(Enum):
    NORMAL = "normal"
    TIME_SENSITIVE = "time_sensitive"
    EMOTIONALLY_SIGNIFICANT = "emotionally_significant"

class CustomerProfile:
    def __init__(self, 
                 customer_id: str,
                 cultural_dimensions: Dict[str, float],
                 emotional_state: Dict[str, float]):
        self.customer_id = customer_id
        self.cultural_dimensions = cultural_dimensions
        self.emotional_state = emotional_state
        self.arrival_time: Optional[datetime] = None
        self.order_type: Optional[OrderType] = None
        self.items: List[Dict] = []
        
    def analyze_emotional_context(self) -> Dict[str, float]:
        """Analyze emotional context based on order and timing"""
        if not self.order_type or not self.items:
            return self.emotional_state
            
        # Check for emotionally significant items
        emotional_items = ["cake", "ice cream", "birthday", "anniversary"]
        for item in self.items:
            if any(keyword in item["name"].lower() for keyword in emotional_items):
                self.emotional_state["excitement"] = min(1.0, self.emotional_state.get("excitement", 0) + 0.3)
                self.emotional_state["impatience"] = min(1.0, self.emotional_state.get("impatience", 0) + 0.2)
                self.order_type = OrderType.EMOTIONALLY_SIGNIFICANT
                
        # Check for time sensitivity
        if self.arrival_time and datetime.now() - self.arrival_time > timedelta(minutes=10):
            self.emotional_state["frustration"] = min(1.0, self.emotional_state.get("frustration", 0) + 0.4)
            self.emotional_state["patience"] = max(0.0, self.emotional_state.get("patience", 1) - 0.3)
            
        return self.emotional_state

class StoreSimulation:
    def __init__(self):
        self.customers: Dict[str, CustomerProfile] = {}
        self.available_associates = 2  # Number of available store associates
        self.queue: List[Tuple[str, datetime]] = []  # (customer_id, join_time)
        
    def customer_arrival(self, customer: CustomerProfile, items: List[Dict]):
        """Handle customer arrival event"""
        customer.arrival_time = datetime.now()
        customer.items = items
        customer.analyze_emotional_context()
        self.customers[customer.customer_id] = customer
        self.queue.append((customer.customer_id, datetime.now()))
        
        # Determine response based on emotional state and cultural profile
        return self.generate_response(customer)
        
    def generate_response(self, customer: CustomerProfile) -> Dict:
        """Generate appropriate response based on customer context"""
        response = {"customer_id": customer.customer_id}
        
        # Check for emotionally significant orders
        if customer.order_type == OrderType.EMOTIONALLY_SIGNIFICANT:
            response["message"] = self._generate_emotional_response(customer)
            response["priority"] = "high"
            response["suggested_actions"] = ["expedite_order", "offer_compassion"]
            
        # Check for long wait times
        elif (datetime.now() - customer.arrival_time) > timedelta(minutes=5):
            response["message"] = "We're preparing your order and will be with you shortly. " \
                                "As a token of our appreciation, here's 10% off your next purchase!"
            response["priority"] = "medium"
            response["suggested_actions"] = ["offer_discount", "update_wait_time"]
            
        # Default response
        else:
            response["message"] = "Thank you for your order! We're preparing it now."
            response["priority"] = "normal"
            response["suggested_actions"] = ["acknowledge"]
            
        return response
        
    def _generate_emotional_response(self, customer: CustomerProfile) -> str:
        """Generate culturally and emotionally appropriate response"""
        if customer.cultural_dimensions.get("formality", 0.5) > 0.7:
            return "We understand this is a special occasion. Your order is being handled with the utmost care and will be ready shortly."
        else:
            return "We can see this is for something special! We're putting a rush on your order. The team is excited to help make this moment wonderful for you!"

def run_simulation():
    """Run a demo of the pickup simulation"""
    print("🚀 Starting Delight.AI Pickup Simulation\n")
    
    # Initialize simulation
    simulation = StoreSimulation()
    
    # Create a customer with an emotionally significant order
    customer = CustomerProfile(
        customer_id="cust123",
        cultural_dimensions={
            "individualism": 0.8,
            "uncertainty_avoidance": 0.6,
            "formality": 0.4
        },
        emotional_state={
            "happiness": 0.7,
            "stress": 0.3,
            "patience": 0.8
        }
    )
    
    # Customer places an order with an ice cream cake
    items = [{"id": "item001", "name": "Chocolate Ice Cream Cake", "quantity": 1}]
    
    print("👋 Customer arrives for pickup")
    print(f"📦 Order contains: {', '.join([item['name'] for item in items])}")
    
    # Process customer arrival
    response = simulation.customer_arrival(customer, items)
    
    print("\n💬 System Response:")
    print(f"Message: {response['message']}")
    print(f"Priority: {response['priority'].upper()}")
    print(f"Suggested Actions: {', '.join(response['suggested_actions'])}")
    
    # Simulate delay
    print("\n⏳ 10 minutes later...")
    customer.arrival_time = datetime.now() - timedelta(minutes=10)
    customer.emotional_state["patience"] = 0.3
    customer.emotional_state["frustration"] = 0.7
    
    # Get updated response
    response = simulation.generate_response(customer)
    
    print("\n💬 Updated Response:")
    print(f"Message: {response['message']}")
    print(f"Priority: {response['priority'].upper()}")
    print(f"Suggested Actions: {', '.join(response['suggested_actions'])}")

if __name__ == "__main__":
    run_simulation()
