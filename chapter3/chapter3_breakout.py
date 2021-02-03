import queue
import random
import threading
from collections import deque

import gym
import numpy
import tensorflow as tf
from pynput.keyboard import Key, Listener
from tensorflow import keras
from tensorflow.keras import layers

from utils import cv2_resize_image


class ReplayMemory:
    def __init__(self, history_len=4, capacity=100000, batch_size=32, input_scale=255.0):
        self.capacity = capacity
        self.history_length = history_len
        self.batch_size = batch_size
        self.input_scale = input_scale

        self.frames = deque([])
        self.others = deque([])

    def add(self, frame, action, r, termination):
        if len(self.frames) == self.capacity:
            self.frames.popleft()
            self.others.popleft()
        self.frames.append(frame)
        self.others.append((action, r, termination))

    def add_nullops(self, init_frame):
        for _ in range(self.history_length):
            self.add(init_frame, 0, 0, 0)

    def phi(self, new_frame):
        assert len(self.frames) > self.history_length
        images = [new_frame] + [self.frames[-1 - i] for i in range(self.history_length - 1)]
        return numpy.concatenate(images, axis=0)

    def _phi(self, index):
        images = [self.frames[index - i] for i in range(self.history_length)]
        return numpy.concatenate(images, axis=0)

    def sample(self):
        while True:
            index = random.randint(a=self.history_length - 1, b=len(self.frames) - 2)
            infos = [self.others[index - i] for i in range(self.history_length)]

            # Check if termination=1 before "index"
            flag = False
            for i in range(1, self.history_length):
                if infos[i][2] == 1:
                    flag = True
                    break
            if flag:
                continue

            state = self._phi(index)
            new_state = self._phi(index + 1)
            action, r, termination = self.others[index]
            state = numpy.asarray(state / self.input_scale, dtype=numpy.float32)
            new_state = numpy.asarray(new_state / self.input_scale, dtype=numpy.float32)
            return (state, action, r, new_state, termination)


class QNetwork:
    def __init__(self, name="q_network", input_shape=(84, 84, 4), n_outputs=4):
        self.name = name

        self.width = input_shape[0]
        self.height = input_shape[1]
        self.channel = input_shape[2]
        self.n_outputs = n_outputs

        self.inputs = layers.Input(shape=(self.width, self.height, self.channel))

        self.cnn1 = layers.Conv2D(32, 8, strides=4, activation="relu")(self.inputs)
        self.cnn2 = layers.Conv2D(64, 4, strides=2, activation="relu")(self.cnn1)
        self.cnn3 = layers.Conv2D(64, 3, strides=1, activation="relu")(self.cnn2)

        self.flatten = layers.Flatten()(self.cnn3)

        self.fc = layers.Dense(512, activation="relu")(self.flatten)
        self.action = layers.Dense(n_outputs, activation="linear")(self.fc)

        self.model = keras.Model(inputs=self.inputs, outputs=self.action)

    def get_model(self):
        return self.model


class Game:
    def __init__(self, name, lost_life_as_terminal=False):
        self.ale = gym.make(name)
        frame = self.ale.reset()
        self.lost_life_as_terminal = lost_life_as_terminal
        self.lives = 0
        self.actions = list(range(self.ale.action_space.n))

        self.frame_skip = 4
        self.total_reward = 0
        self.crop_size = 84
        self.crop_offset = 8

        # Frame buffer
        self.buffer_size = 8
        self.buffer_index = 0
        self.buffer = [self.crop(self.rgb_to_gray(frame)) for _ in range(self.buffer_size)]
        # Overlapping frames, maximum of two frames
        self.last_frame = frame
        self.ale.render()

    def set_params(self, crop_size=84, crop_offset=8, frame_skip=4, lost_life_as_terminal=False):
        self.crop_size = crop_size
        self.crop_offset = crop_offset
        self.frame_skip = frame_skip
        self.lost_life_as_terminal = lost_life_as_terminal

        frame = self.ale.reset()
        self.buffer = [self.crop(self.rgb_to_gray(frame)) for _ in range(self.buffer_size)]
        self.last_frame = frame

    def crop(self, frame):
        feedback = cv2_resize_image(frame, resized_shape=(self.crop_size, self.crop_size), method="crop", crop_offset=self.crop_offset)
        return feedback

    @staticmethod
    def rgb_to_gray(im):
        return numpy.dot(im, [0.2126, 0.7152, 0.0722])

    def get_current_feedback(self, num_frames=1):
        assert num_frames < self.buffer_size, "Frame buffer is not large enough."
        index = self.buffer_index - 1
        frames = [numpy.expand_dims(self.buffer[index - k], axis=0) for k in range(num_frames)]
        if num_frames > 1:
            return numpy.concatenate(frames, axis=0)
        else:
            return frames[0]

    def get_total_reward(self):
        return self.total_reward

    def add_frame_to_buffer(self, frame):
        self.buffer_index = self.buffer_index % self.buffer_size
        self.buffer[self.buffer_index] = frame
        self.buffer_index += 1

    def _lost_life(self, info):
        if self.lost_life_as_terminal:
            lives = info["ale.lives"]
            if lives >= self.lives:
                self.lives = lives
                return False
            else:
                return True
        else:
            return False

    def reset(self):
        frame = self.ale.reset()
        self.total_reward = 0
        self.buffer_index = 0
        self.lives = 0
        self.buffer = [self.crop(self.rgb_to_gray(frame)) for _ in range(self.buffer_size)]
        self.last_frame = frame

    def get_available_actions(self):
        return list(range(len(list(range(self.ale.action_space.n)))))

    def play_action(self, action, num_frames=1):
        reward = 0
        termination = 0
        for i in range(self.frame_skip):
            a = self.actions[action]
            frame, r, done, info = self.ale.step(a)
            self.ale.render()
            reward += r
            if i == self.frame_skip - 2:
                self.last_frame = frame
            if done or self._lost_life(info):
                termination = 1
        self.add_frame_to_buffer(self.crop(numpy.maximum(self.rgb_to_gray(frame), self.rgb_to_gray(self.last_frame))))

        r = numpy.clip(reward, -1, 1)
        self.total_reward += reward

        return self.get_current_feedback(num_frames), r, termination


def keyboard(queue):
    def on_press(key):
        if key == Key.esc:
            queue.put(-1)
        elif key == Key.space:
            queue.put(ord(" "))
        else:
            key = str(key).replace("'", "")
            if key in ["w", "a", "s", "d"]:
                queue.put(ord(key))

    def on_release(key):
        if key == Key.esc:
            return False

    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def start_game(queue):
    game = Game("BreakoutNoFrameskip-v4")
    game.set_params(frame_skip=4, lost_life_as_terminal=False)

    key_to_act = game.ale.env.get_keys_to_action()
    key_to_act = {k[0]: a for k, a in key_to_act.items() if len(k) > 0}

    n_outputs = len(game.get_available_actions())

    q_network = QNetwork(name="q_network", n_outputs=n_outputs)
    target_network = QNetwork(name="target_network", n_outputs=n_outputs)
    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

    max_episodes = 500000

    num_of_trials = -1

    epsilon_min = 0.1
    epsilon_decay = 1000000
    seed = 42
    gamma = 0.99
    batch_size = 32

    # Experience replay buffers
    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    frame_count = 0

    max_memory_length = 500000

    update_after_actions = 4
    update_target_network = 10000

    feedback_size = (84, 84)

    replay_memory = ReplayMemory()
    loss_function = keras.losses.Huber()

    for episode in range(max_episodes):
        game.reset()
        frame = game.get_current_feedback()

        for _ in range(5):
            next_frame, reward, termination = game.play_action(action=0)
            replay_memory.add(frame, 0, reward, termination)
            frame = next_frame

        for _ in range(100000):
            num_of_trials += 1

            state = replay_memory.phi(frame)

            # Decay probability of taking random action
            epsilon = epsilon_min + max(epsilon_decay - num_of_trials, 0) / epsilon_decay * (1 - epsilon_min)
            print(f"epi {episode}, frame {int(num_of_trials / 1000)}k: reward {game.get_total_reward()}, eps {epsilon}", end="\r")

            if numpy.random.binomial(1, epsilon) == 1:
                action = random.choice(game.get_available_actions())
            else:
                # Predict action Q-values
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, axis=0)
                state_tensor = tf.transpose(state_tensor, perm=(0, 2, 3, 1))
                action_probs = q_network.get_model()(state_tensor, training=False)
                action = tf.argmax(action_probs[0]).numpy()

            next_frame, reward, termination = game.play_action(action)
            replay_memory.add(frame, action, reward, termination)
            frame = next_frame

            if num_of_trials % update_after_actions == 0 and num_of_trials > batch_size:

                w, h = feedback_size

                state_sample = numpy.zeros((batch_size, w, h, 4), dtype=numpy.float32)
                state_next_sample = numpy.zeros((batch_size, w, h, 4), dtype=numpy.float32)
                rewards_sample = numpy.zeros(batch_size, dtype=numpy.float32)
                action_sample = numpy.zeros(batch_size, dtype=numpy.int32)
                done_sample = numpy.zeros(batch_size, dtype=numpy.int32)

                for i in range(batch_size):
                    state, action, reward, new_state, termination = replay_memory.sample()

                    state = numpy.transpose(state, (1, 2, 0))
                    new_state = numpy.transpose(new_state, (1, 2, 0))

                    state_sample[i] = state
                    state_next_sample[i] = new_state
                    rewards_sample[i] = reward
                    action_sample[i] = action
                    done_sample[i] = termination

                # Build the updated Q-values for the sampled future states
                # Use the target model for stability
                future_rewards = target_network.get_model().predict(state_next_sample)

                # Q value = reward + discount factor * expected future reward
                updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)
                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                # Create a mask so we only calculate loss on the updated Q-values
                masks = tf.one_hot(action_sample, len(game.get_available_actions()))

                with tf.GradientTape() as tape:
                    # Train the model on the states and updated Q-values
                    q_values = q_network.get_model()(state_sample)

                    # Apply the masks to the Q-values to get the Q-value for action taken
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    # Calculate loss between new Q-value and old Q-value
                    loss = loss_function(updated_q_values, q_action)

                # Backpropagation
                grads = tape.gradient(loss, q_network.get_model().trainable_variables)
                optimizer.apply_gradients(zip(grads, q_network.get_model().trainable_variables))

            if num_of_trials % update_target_network == 0:
                # update the the target network with new weights
                target_network.get_model().set_weights(q_network.get_model().get_weights())

            if termination:
                print(f"epi {episode}, frame {int(num_of_trials / 1000)}k: reward {game.get_total_reward()}, eps {epsilon}")
                break


if __name__ == "__main__":
    queue = queue.Queue(maxsize=10)
    game_thread = threading.Thread(target=start_game, args=(queue,))
    game_thread.start()

    keyboard(queue)
