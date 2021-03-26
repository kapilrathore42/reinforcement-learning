import gym
import tensorflow as tf 
import numpy as np
from tensorflow.keras import  layers
from tensorflow import keras
num_states = 2
num_actions = 3
action_probs = []
action = []
    
def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(num_states))
    x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
    actions = tf.keras.layers.Dense(num_actions, activation=tf.nn.softmax)(x)
    return keras.Model(inputs=inputs, outputs=actions)
    # Convolutions on the frames on the screen

# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model()
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model()
nextstep_reward = []
reward = []
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
gamma = 0.99
batch_size = 32
# Number of frames to take random action and observe output
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 10000
# Using huber loss for stability
loss_function = keras.losses.Huber()

env = gym.make("MountainCar-v0")
for i_episode in range(100):
    state = np.array(env.reset())
    episode_reward = 0
    for step in range(10000):
        env.render()
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        #action = model(state_tensor, training=False)
        action_probs = model(state_tensor, training=False)
            # Take best action
        action = tf.argmax(action_probs[0]).numpy()
            # Take best action
        state_next , reward ,done,info = env.step(action)
        state_next = np.array(state_next)
        episode_reward += reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next
            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
        indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
        state_sample = np.array([state_history[i] for i in indices])
        state_next_sample = np.array([state_next_history[i] for i in indices])
        rewards_sample = [rewards_history[i] for i in indices]
        action_sample = [action_history[i] for i in indices]
        done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
        future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
        updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample, num_actions)

        with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        model_target.set_weights(model.get_weights())
            # Log details
       # template = "running reward: {:.2f} at episode {}"
       # print(template.format(running_reward, episode_count))
    episode_reward_history.append(episode_reward)
    running_reward = np.mean(episode_reward_history)
    episode_count += 1
    template = "running reward: {:.2f} at episode {}"
    print(template.format(running_reward, episode_count))
    env.close()