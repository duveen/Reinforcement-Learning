import gym
import numpy as np
import random
import math

# 라이브러리 불러오기
environment = gym.make("CartPole-v0")
# gym 라이브러리에서 CartPole 객체 생성

no_buckets = (1, 1, 6, 3)

no_actions = environment.action_space.n

state_value_bounds = list(
    zip(environment.observation_space.low, environment.observation_space.high)
)
state_value_bounds[1] = [-0.5, 0.5]
state_value_bounds[3] = [-math.radians(50), math.radians(50)]
action_index = len(no_buckets)

q_value_table = np.zeros(no_buckets + (no_actions,))
# 현재 좌표값 테이블 배열 생성
min_explore_rate = 0.01
min_learning_rate = 0.1
# 학습에 필요한 하이퍼 파라미터
max_episodes = 1000
# 최대 탐험 횟수
max_time_steps = 250
# 최대 시간당 할수 있는 횟수
streak_to_end = 120
solved_time = 199
discount = 0.99
no_streaks = 0
# 하이퍼 파라미터 설정


def select_action(state_value, explore_rate):
    if random.random() < explore_rate:
        action = environment.action_space.sample()
    else:
        action = np.argmax(q_value_table[state_value])
    return action


# pole이 움직이게 하는 메소드


def select_explore_rate(x):
    return max(min_explore_rate, min(1, 1.0 - math.log10((x + 1) / 25)))


def select_learning_rate(x):
    return max(min_learning_rate, min(0.5, 1.0 - math.log10((x + 1) / 25)))


# 초기 학습 값 설정


def bucketize_state_value(state_value):
    bucket_indexes = []
    for i in range(len(state_value)):
        if state_value[i] <= state_value_bounds[i][0]:
            bucket_index = 0
        elif state_value[i] >= state_value_bounds[i][1]:
            bucket_index = no_buckets[i] - 1
        else:
            bound_width = state_value_bounds[i][1] - state_value_bounds[i][0]
            offset = (no_buckets[i] - 1) * state_value_bounds[i][0] / bound_width
            scaling = (no_buckets[i] - 1) / bound_width
            bucket_index = int(round(scaling * state_value[i] - offset))
        bucket_indexes.append(bucket_index)
    return tuple(bucket_indexes)


# 여기는 좌표값 설정하는 부분

for episode_no in range(max_episodes):
    explore_rate = select_explore_rate(episode_no)
    learning_rate = select_learning_rate(episode_no)

    observation = environment.reset()

    # 초기 관찰 값 및 초기 rate 설정
    start_state_value = bucketize_state_value(observation)
    previous_state_value = start_state_value
    # 초기 pole의 움직이는 좌표값 설정
    for time_step in range(max_time_steps):
        environment.render()
        selected_action = select_action(previous_state_value, explore_rate)
        observation, reward_gain, completed, _ = environment.step(selected_action)
        # 여기서 초기값으로 인한 관측 값, 보상값, 성공값을 받음
        state_value = bucketize_state_value(observation)
        # 관측값으로 인해 지금 현재값을 얻고
        best_q_value = np.amax(q_value_table[state_value])

        q_value_table[previous_state_value + (selected_action,)] += learning_rate * (
            reward_gain
            + discount * (best_q_value)
            - q_value_table[previous_state_value + (selected_action,)]
        )
        # 움직였을때 최고로 좋은 좌표값을 best_q_value로 얻음으로써 업데이트를 계속 해준다
        print("Episode number : %d" % episode_no)
        print("Time step : %d" % time_step)
        print("Selection action : %d" % selected_action)
        print("Current state : %s" % str(state_value))
        print("Reward obtained : %f" % reward_gain)
        print("Best Q value : %f" % best_q_value)
        print("Learning rate : %f" % learning_rate)
        print("Explore rate : %f" % explore_rate)
        print("Streak number : %d" % no_streaks)

        if completed:
            print("Episode %d finished after %f time steps" % (episode_no, time_step))
            if time_step >= solved_time:
                no_streaks += 1
            else:
                no_streaks = 0
            break

        previous_state_value = state_value
    # if문을통해 일정 값이 넘으면 성공했다고 count를 해줌
    if no_streaks > streak_to_end:
        break
    # 만약 성공값이 미리 설정해놓은 end값보다 높으면 학습이 잘 되었다 판단하여 마무리
