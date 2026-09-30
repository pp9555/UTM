import numpy as np
import websocket
import json
import time

# Simulation Parameters
STRAIN_MAX = 0.2
POINTS = 200
ELASTIC_LIMIT_STRAIN = 0.05
YOUNGS_MODULUS = 2000 # Fake MPa
NOISE_STD_DEV = 1.5

def generate_fake_curve():
    strains = np.linspace(0, STRAIN_MAX, POINTS)
    data_points = []
    
    for x in strains:
        # Elastic Region (Linear)
        if x <= ELASTIC_LIMIT_STRAIN:
            y = x * YOUNGS_MODULUS
        # Fracture Point
        elif x > 0.18:
            y = 0
        # Plastic Region (Logarithmic/Yielding)
        else:
            yield_stress = ELASTIC_LIMIT_STRAIN * YOUNGS_MODULUS
            y = yield_stress + 20 * np.log(1 + 50 * (x - ELASTIC_LIMIT_STRAIN))
            
        # Add realistic sensor noise
        if y > 0:
            y += np.random.normal(0, NOISE_STD_DEV)
            
        data_points.append({"strain": round(x, 4), "stress": round(y, 2)})
        
    return data_points

def stream_data():
    curve_data = generate_fake_curve()
    
    # Connect to Node.js Relay
    ws = websocket.WebSocket()
    ws.connect("ws://localhost:8080")
    print("Connected to Node.js WebSocket Server")
    
    print("Starting Tensile Test Simulation...")
    for point in curve_data:
        payload = json.dumps(point)
        ws.send(payload)
        print(f"Sent: {payload}")
        time.sleep(0.1) # Simulate the physical machine slowly pulling (10Hz)
        
    ws.close()
    print("Test Complete. Specimen Fractured.")

if __name__ == "__main__":
    stream_data()