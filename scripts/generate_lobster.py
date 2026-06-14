import csv
import random

def generate_synthetic_lobster(filename, num_messages=100000):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        timestamp = 34200.0  # 9:30 AM in seconds
        price = 150.0
        order_id = 1
        
        for _ in range(num_messages):
            msg_type = random.choices([1, 2, 3, 4, 5], weights=[40, 20, 20, 15, 5])[0]
            size = random.randint(10, 1000)
            
            # Random walk for price
            if random.random() > 0.5:
                price += random.choice([0.01, -0.01])
            price = round(price, 2)
            
            direction = random.choice([1, -1])
            
            writer.writerow([f"{timestamp:.6f}", msg_type, order_id, size, price, direction])
            
            timestamp += random.uniform(0.00001, 0.001)  # microsecond increments
            order_id += 1

if __name__ == "__main__":
    import sys
    filename = sys.argv[1] if len(sys.argv) > 1 else "lobster_sample.csv"
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    generate_synthetic_lobster(filename, num)
    print(f"Generated {num} synthetic messages to {filename}")
