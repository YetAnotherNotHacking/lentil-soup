import requests
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# Configuration
API_URL = 'http://localhost:3000'
ADDRESS = 'sf-host-test.play.minekube.net'
PORT = 25565
EMAIL = 'get your own account neeeeeeeeeeeeeeeeeeeeeeeeeerd'
AUTH = 'microsoft'

# Constants
REINFORCEMENT_INTERVAL = 5  # Time in seconds for retraining
LEARNING_RATE = 0.35  # Learning rate for the Q-learning model
GAMMA = 0.70  # Discount factor for future rewards
BATCH_SIZE = 128  # Number of experiences to sample from replay buffer
REPLAY_MEMORY_SIZE = 200000000  # Maximum size of the replay memory
EPSILON = 0.9  # Exploration rate
EPSILON_MIN = 0.1  # Minimum exploration rate
EPSILON_DECAY = 1.0  # Decay rate for exploration
EXPECTED_STATE_LENGTH = 9  # The length of the state vector from the API response
MOVE_STEP = 20  # Number of blocks to move in one step

# Neural Network for Q-learning
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(EXPECTED_STATE_LENGTH, 32896)
        self.fc2 = nn.Linear(32896, 32896)
        self.fc3 = nn.Linear(32896, 6)  # Number of actions, increase this number if you give the bot more abilities

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, optimizer, and loss function
model = QNetwork()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Experience replay buffer
replay_buffer = deque(maxlen=REPLAY_MEMORY_SIZE)

def safe_request(method, url, **kwargs):
    max_retries = 2
    for attempt in range(max_retries):
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)
    print("Max retries exceeded.")
    return None

def login():
    response = safe_request('POST', f'{API_URL}/login', json={
        'address': ADDRESS,
        'port': PORT,
        'account': EMAIL,
        'auth': AUTH
    })
    if response and response.status_code == 200:
        print('Logged in successfully.')
    else:
        print(f'Error logging in: {response.text if response else "No response"}')
        exit(1)

def logout():
    response = safe_request('POST', f'{API_URL}/logout')
    if response and response.status_code == 200:
        print('Logged out successfully.')
    else:
        print(f'Error logging out: {response.text if response else "No response"}')

def perform_action(action):
    status = get_status()
    if not status:
        return False
    
    x, y, z = status['position']['x'], status['position']['y'], status['position']['z']

    if action == 0:  # Move forward
        response = safe_request('POST', f'{API_URL}/move', json={'x': x + MOVE_STEP, 'y': y, 'z': z})
        success = response and response.status_code == 200
        print(f'Move forward action: {success}, Response: {response.text if response else "No response"}')
        return success
    elif action == 1:  # Move backward
        response = safe_request('POST', f'{API_URL}/move', json={'x': x - MOVE_STEP, 'y': y, 'z': z})
        success = response and response.status_code == 200
        print(f'Move backward action: {success}, Response: {response.text if response else "No response"}')
        return success
    elif action == 2:  # Move left
        response = safe_request('POST', f'{API_URL}/move', json={'x': x, 'y': y, 'z': z - MOVE_STEP})
        success = response and response.status_code == 200
        print(f'Move left action: {success}, Response: {response.text if response else "No response"}')
        return success
    elif action == 3:  # Move right
        response = safe_request('POST', f'{API_URL}/move', json={'x': x, 'y': y, 'z': z + MOVE_STEP})
        success = response and response.status_code == 200
        print(f'Move right action: {success}, Response: {response.text if response else "No response"}')
        return success
    elif action == 4:  # Look
        yaw, pitch = random.uniform(0, 360), random.uniform(-90, 90)
        response = safe_request('POST', f'{API_URL}/look', json={'yaw': yaw, 'pitch': pitch})
        success = response and response.status_code == 200
        print(f'Look action: {success}, Response: {response.text if response else "No response"}')
        return success
    elif action == 5: # attack
        response = safe_request('POST', f'{API_URL}/attack')
        success = response and response.status_code == 200
        print(f'Attack action: {success}, Response: {response.text if response else "No response"}')

    # bot really needs a large number of functions, any proposed ideas for how to do this without making this script 10^99 lines long will be appreciated.
        



    print('No valid action performed')
    requests.post(f'{API_URL}/chat', json={'message': 'No valid action performed, waiting for network to choose another action...'})
    return False

def get_status():
    response = safe_request('GET', f'{API_URL}/status')
    if response and response.status_code == 200:
        status = response.json()
        print(f'Fetched status: {status}')
        return status
    else:
        print(f'Error fetching status: {response.text if response else "No response"}')
        return None

def get_inventory():
    response = safe_request('GET', f'{API_URL}/inventory')
    if response and response.status_code == 200:
        inventory = response.json()
        print(f'Fetched inventory: {inventory}')
        return inventory
    else:
        print(f'Error fetching inventory: {response.text if response else "No response"}')
        return None

def find_mobs(status):
    mobs = []
    if 'entities' in status:
        for entity in status['entities']:
            if entity['type'] == 'mob':
                mobs.append(entity)
                print(f"Found mob: {entity}")
    return mobs

last_log_time = time.time()

def log_status_and_inventory():
    global last_log_time
    current_time = time.time()
    if current_time - last_log_time >= 10:  # Adjust for how long between the information available to the bot is refreshed.
        status = get_status()
        inventory = get_inventory()
        print(f"Fetched status: {status}")
        print(f"Fetched inventory: {inventory}")
        print(f"Mobs found: {find_mobs(status)}")
        last_log_time = current_time

def choose_action(state):
    global EPSILON
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)
    if random.random() < EPSILON: # Exploration.
        action = random.randint(0, 4)  # Random action (0 to 4)
    else:
        action = torch.argmax(q_values).item()
    print(f'Chosen action: {action}')
    return action

def update_replay_buffer(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))

def what_is_action(actionid):
    if actionid == 0:
        return "Move Forward"
    elif actionid == 1:
        return "Move Backward"
    elif actionid == 2:
        return "Move Left"
    elif actionid == 3:
        return "Move Right"
    elif actionid == 4:
        return "Look"
    elif actionid == 5:
        return "Attack"
    else:
        return "Invalid Action"

def train_model():
    requests.post(f'{API_URL}/chat', json={'message': 'Training start, applying Q-learning...'})
    if len(replay_buffer) < BATCH_SIZE:
        return
    
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.long)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
    dones_tensor = torch.tensor(dones, dtype=torch.float32)

    current_q_values = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
    next_q_values = model(next_states_tensor).max(1)[0]
    target_q_values = rewards_tensor + (GAMMA * next_q_values * (1 - dones_tensor))

    loss = criterion(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    global EPSILON
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    # Save model
    requests.post(f'{API_URL}/chat', json={'message': 'Training finished, now saving model, thread might stop...'})
    torch.save(model.state_dict(), 'model.pth')

def main():
    login()
    last_retraining_time = time.time()

    while True:
        log_status_and_inventory()

        status = get_status()
        if not status:
            break

        inventory = get_inventory()
        state = [
            status['health'], status['position']['x'], status['position']['y'], status['position']['z'],
            status['rotation']['yaw'], status['rotation']['pitch'], status['saturation'],
            len(find_mobs(status)), len(inventory)
        ]

        if len(state) != EXPECTED_STATE_LENGTH:
            print(f"State length mismatch: {len(state)}")
            continue

        action = choose_action(state)
        # requests.post(f'{API_URL}/chat', json={'message': f'Performing action: {what_is_action(action)}'})
        success = perform_action(action)
        reward = calculate_reward(success, action, inventory)
        next_status = get_status()
        if next_status:
            next_inventory = get_inventory()
            next_state = [
                next_status['health'], next_status['position']['x'], next_status['position']['y'], next_status['position']['z'],
                next_status['rotation']['yaw'], next_status['rotation']['pitch'], next_status['saturation'],
                len(find_mobs(next_status)), len(next_inventory)
            ]
        else:
            next_state = state
        done = next_status is None
        update_replay_buffer(state, action, reward, next_state, done)

        current_time = time.time()
        if current_time - last_retraining_time >= REINFORCEMENT_INTERVAL:
            # requests.post(f'{API_URL}/chat', json={'message': 'Retraining the model...'})
            train_model()
            # save the model
            torch.save(model.state_dict(), 'model.pth')
            # requests.post(f'{API_URL}/chat', json={'message': 'Model saved'})
            # find the overall reward
            total_reward = 0
            for experience in replay_buffer:
                total_reward += experience[2]
            # requests.post(f'{API_URL}/chat', json={'message': f'Total reward: {total_reward}'})
            # say in chat about retraining
            last_retraining_time = current_time

        time.sleep(1)
    
    logout()

def calculate_reward(success, action, inventory):
    # Get status of the player
    status = get_status()
    if not status:
        return 0
    
    mobs = find_mobs(status)
    # Current position of player:
    x, y, z = status['position']['x'], status['position']['y'], status['position']['z']

    # Get a 5 block radius of block data around the player
    blocks = []

    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                response = safe_request('GET', f'{API_URL}/block', params={'x': x + i, 'y': y + j, 'z': z + k})
                if response and response.status_code == 200:
                    blocks.append(response.json())
                else:
                    print(f'Error fetching block data: {response.text if response else "No response"}')
                    return 0

# vvv Please fix!!!! vvv
    
#    # Penalty for being in water
#    for block in blocks:
#        if block['name'] == 'water':
#            reward -= 1

    if action == 4:  # Use Item action
        if inventory:
            reward += 1

    # Reward for getting more blocks
    if inventory:
        for item in inventory:
            if item['name'] == 'slime':
                reward += 0.5
            elif item['name'] == 'dirt':
                reward -= 0.32
    
    # Large penalty for dying
    if not success:
        reward -= 100

    # Penalty for losing health
    if success and action in {0, 1, 2, 3}:
        reward -= 0.1 * (20 - inventory['health'])

    # Reward for mining ores
    for item in inventory:
        if item['name'] in {'diamond', 'gold', 'iron', 'coal'}:
            reward += 14.889
        else:
            reward -= 0.01
    
    # Reward for getting wood, any type
    for item in inventory:
        if item['name'] in {'oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log', 'dark_oak_log'}:
            reward += 0.5
        else:
            reward -= 0.01

    # Reward for getting food
    for item in inventory:
        if item['name'] in {'cooked_beef', 'cooked_chicken', 'cooked_porkchop', 'cooked_mutton', 'cooked_rabbit', 'baked_potato', 'bread'}:
            reward += 0.5
        else:
            reward -= 0.01
    
    # Reward for getting tools
    for item in inventory:
        if item['name'] in {'netherite_pickaxe','diamond_pickaxe', 'iron_pickaxe', 'stone_pickaxe', 'wooden_pickaxe'}:
            reward += 0.5
        else:
            reward -= 0.01

    # Medium penalty for standing still
    if action == 4:
        reward -= 0.5

    # Reward for being near ores
    for block in blocks:
        if block['name'] in {'diamond_ore', 'gold_ore', 'iron_ore', 'coal_ore'}:
            reward += 0.5

    # Reward for having experience orbs entity close by
    for mob in mobs:
        if mob['name'] == 'experience_orb':
            reward += 0.5
    
    # Reward for having more XP than before
    if status['experience'] > status['experience'] + 1:
        reward += 0.5

    # Reward for attacking mobs
    if action == 5:
        reward += 2


    elif action in {0, 1, 2, 3}:  # Move actions
        if success:
            reward += 0.5  # Partial reward for successful movement
        else:
            reward -= 0.5  # Penalty for failed movement
    else:
        reward -= 1  # Penalty for invalid actions
    return reward

if __name__ == "__main__":
    main()
