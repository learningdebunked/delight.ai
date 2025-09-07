"""
Delight.AI - Queue Management Simulation

This module simulates queue management scenarios to demonstrate how Delight.AI
can improve customer experience during peak times and long waits.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
import random
import time
import math

class CustomerMood(Enum):
    CALM = "calm"
    IMPATIENT = "impatient"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"

class Customer:
    def __init__(self, customer_id: str, arrival_time: datetime):
        self.customer_id = customer_id
        self.arrival_time = arrival_time
        self.mood = CustomerMood.CALM
        self.wait_time = timedelta(0)
        # Base patience between 3-8 minutes, with some customers being more patient
        self.base_patience = random.uniform(3, 8) * (0.5 if random.random() < 0.2 else 1.2)
        self.last_mood_change = arrival_time
        
    def update_mood(self, current_time: datetime):
        """Update customer's mood based on wait time and queue conditions"""
        self.wait_time = current_time - self.arrival_time
        wait_minutes = self.wait_time.total_seconds() / 60
        
        # Calculate dynamic patience based on queue movement
        time_since_last_change = (current_time - self.last_mood_change).total_seconds() / 60
        
        # Only allow mood to change every 30 seconds
        if time_since_last_change < 0.5:
            return
            
        # Update mood based on wait time
        if wait_minutes > self.base_patience * 1.5:
            if self.mood != CustomerMood.ANGRY:
                self.last_mood_change = current_time
                self.mood = CustomerMood.ANGRY
        elif wait_minutes > self.base_patience * 1.2:
            if self.mood != CustomerMood.FRUSTRATED:
                self.last_mood_change = current_time
                self.mood = CustomerMood.FRUSTRATED
        elif wait_minutes > self.base_patience * 0.8:
            if self.mood != CustomerMood.IMPATIENT:
                self.last_mood_change = current_time
                self.mood = CustomerMood.IMPATIENT
            
class QueueSimulation:
    def __init__(self, num_counters: int = 3):
        self.queues = [[] for _ in range(num_counters)]
        self.counters = [None] * num_counters  # Currently serving customer
        self.customer_counter = 0
        self.customers: Dict[str, Customer] = {}
        self.start_time = datetime.now()
        
    def add_customer(self):
        """Add a new customer to the shortest queue"""
        self.customer_counter += 1
        customer_id = f"cust_{self.customer_counter}"
        customer = Customer(customer_id, datetime.now())
        self.customers[customer_id] = customer
        
        # Find the shortest queue
        shortest_q = min(range(len(self.queues)), key=lambda i: len(self.queues[i]))
        self.queues[shortest_q].append(customer_id)
        
        return customer_id, shortest_q
    
    def process_customers(self):
        """Process customers at each counter"""
        current_time = datetime.now()
        
        for i in range(len(self.counters)):
            # If counter is free and queue is not empty
            if self.counters[i] is None and self.queues[i]:
                customer_id = self.queues[i].pop(0)
                # More experienced staff at counter 0 (faster processing)
                base_time = 2.0 if i == 0 else 3.0
                self.counters[i] = {
                    'customer_id': customer_id,
                    'start_time': current_time,
                    'processing_time': max(0.5, random.normalvariate(base_time, 0.5))  # Minutes
                }
                print(f"✅ Counter {i+1} started serving {customer_id}")
            
            # Process current customer
            if self.counters[i] is not None:
                customer = self.counters[i]
                elapsed = (current_time - customer['start_time']).total_seconds() / 60
                
                if elapsed >= customer['processing_time']:
                    # Customer processed
                    cust_id = customer['customer_id']
                    wait_time = (customer['start_time'] - self.customers[cust_id].arrival_time).total_seconds() / 60
                    print(f"🏁 {cust_id} served at Counter {i+1} after {wait_time:.1f} minutes")
                    del self.customers[cust_id]
                    self.counters[i] = None
    
    def get_queue_analytics(self):
        """Get current queue statistics"""
        current_time = datetime.now()
        mood_distribution = {mood: 0 for mood in CustomerMood}
        total_wait = timedelta(0)
        
        for customer in self.customers.values():
            customer.update_mood(current_time)
            mood_distribution[customer.mood] += 1
            total_wait += customer.wait_time
        
        avg_wait = total_wait / len(self.customers) if self.customers else timedelta(0)
        
        return {
            'total_customers': len(self.customers),
            'mood_distribution': {k.value: v for k, v in mood_distribution.items()},
            'average_wait': avg_wait,
            'queue_lengths': [len(q) for q in self.queues]
        }
    
    def generate_mitigation_strategy(self, analytics: dict):
        """Generate strategies to improve queue experience"""
        strategies = []
        
        if analytics['average_wait'] > timedelta(minutes=10):
            strategies.append("Open additional checkout counters")
            strategies.append("Deploy mobile checkout assistants")
            
        if analytics['mood_distribution'].get('frustrated', 0) > 2:
            strategies.append("Offer complimentary refreshments to waiting customers")
            
        if analytics['mood_distribution'].get('angry', 0) > 0:
            strategies.append("Manager intervention required")
            strategies.append("Offer express checkout for angry customers")
            
        # Dynamic queue management
        max_queue_diff = max(analytics['queue_lengths']) - min(analytics['queue_lengths'])
        if max_queue_diff > 2:
            strategies.append("Implement single queue with multiple servers")
            
        return strategies if strategies else ["Queue conditions are optimal"]

def run_queue_simulation(duration_minutes=3):
    """Run the queue management simulation
    
    Args:
        duration_minutes: How long to run the simulation (default: 3 minutes)
    """
    print("🛒 Starting Delight.AI Queue Management Simulation")
    print(f"⏱️  Running for {duration_minutes} minutes...\n")
    print("🏪 Store Information:")
    print("- 3 Checkout Counters")
    print("- Counter 1: Experienced Staff (faster)")
    print("- Counters 2-3: Regular Staff")
    print("\n🔄 Simulation updates every 30 seconds\n")
    
    simulation = QueueSimulation(num_counters=3)
    start_time = datetime.now()
    
    try:
        last_update = datetime.now()
        while (datetime.now() - start_time).total_seconds() < duration_minutes * 60:
            current_time = datetime.now()
            
            # Add new customers in waves
            time_elapsed = (current_time - start_time).total_seconds()
            wave_intensity = 0.4 + 0.3 * (1 + math.sin(time_elapsed / 30))  # Varies between 0.1 and 0.7
            
            if random.random() < wave_intensity * 0.5:  # 20-50% chance to add customer
                cust_id, q_num = simulation.add_customer()
                print(f"👤 Customer {cust_id} joined queue {q_num+1}")
                
                # Update all customers' moods
                for customer in simulation.customers.values():
                    customer.update_mood(current_time)
            
            # Process customers
            simulation.process_customers()
            
            # Get analytics
            analytics = simulation.get_queue_analytics()
            
            # Print status every 30 seconds
            if (current_time - last_update).total_seconds() >= 30:
                last_update = current_time
                analytics = simulation.get_queue_analytics()
                
                print("\n" + "="*60)
                print(f"⏰ Simulation Time: {int(time_elapsed/60)}m {int(time_elapsed%60)}s / {duration_minutes}m")
                print("📊 Queue Status:")
                print(f"👥 Total customers in store: {analytics['total_customers']}")
                print(f"📝 Queue lengths: Counter 1: {analytics['queue_lengths'][0]} | "
                      f"Counter 2: {analytics['queue_lengths'][1]} | "
                      f"Counter 3: {analytics['queue_lengths'][2]}")
                
                # Show customer mood distribution
                moods = analytics['mood_distribution']
                total = sum(moods.values())
                if total > 0:
                    mood_chart = "😊 " * moods.get('calm', 0) + "😐 " * moods.get('impatient', 0) + "😠 " * moods.get('frustrated', 0) + "👿 " * moods.get('angry', 0)
                    print(f"😊 Customer Moods: {mood_chart}")
                
                avg_wait_min = analytics['average_wait'].total_seconds() / 60
                print(f"⏳ Average wait time: {avg_wait_min:.1f} minutes")
                
                # Get and display mitigation strategies
                strategies = simulation.generate_mitigation_strategy(analytics)
                if strategies and strategies[0] != "Queue conditions are optimal":
                    print("\n🚨 Recommended Actions:")
                    for i, strategy in enumerate(strategies, 1):
                        print(f"   {i}. {strategy}")
                
                print("="*60 + "\n")
            
            time.sleep(1)  # Simulate time passing
            
    except KeyboardInterrupt:
        print("\nSimulation ended by user")
    
    print("\n🏁 Simulation complete!")

if __name__ == "__main__":
    run_queue_simulation()
