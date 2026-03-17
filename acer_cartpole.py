import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


### 공통 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENV_NAME = "CartPole-v1"
GAMMA = 0.99

LR = 2e-4
ENTROPY_COEF = 1e-3

# ReplayBuffer 관련
REPLAY_CAPACITY = 50000
BATCH_SIZE = 64
REPLAY_START_SIZE = 1000  # 이만큼 쌓인 뒤부터 off-policy 업데이트 시작

# 중요도 샘플링(Importance Sampling) 클리핑
IS_CLIP = 10.0

MAX_EPISODES = 1000


### Actor-Critic 네트워크
class ActorCritic(nn.Module):
    # CartPole 전용 Actor-Critic 네트워크.
    # - 입력: 상태 (4차원)
    # - 출력:
    #   - 정책 π(a|s): 2차원 softmax 확률
    #   - 상태 가치 V(s): 스칼라

    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.pi_head = nn.Linear(hidden_dim, action_dim)
        self.v_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        logits = self.pi_head(x)
        v = self.v_head(x)
        pi = F.softmax(logits, dim=-1)
        return pi, v.squeeze(-1)  # pi: (N,2), v: (N,)


    def act(self, s: np.ndarray):
        # 단일 상태에서 행동 샘플 + 정책 확률, 가치 반환 (on-policy 수집용)
        with torch.no_grad():
            s_t = torch.as_tensor(s, dtype=torch.float32, device=DEVICE)
            pi, v = self.forward(s_t)
            dist = Categorical(pi)
            a = dist.sample()
            return a.item(), pi.cpu().numpy(), float(v.item())


### Replay Buffer
class ReplayBuffer:
    # (s, a, r, done, s_next, behavior_prob) 를 저장.
    # - behavior_prob: 데이터를 수집할 때 사용한 정책 μ(a|s)의 확률 (Importance Sampling에 필요)
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        done: float,
        s_next: np.ndarray,
        behavior_prob: float,
    ):
        self.buffer.append((s, a, r, done, s_next, behavior_prob))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, done, s_next, mu = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(done, dtype=np.float32),
            np.array(s_next, dtype=np.float32),
            np.array(mu, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


### ACER 스타일 업데이트 함수들
def acer_on_policy_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    s_traj,
    a_traj,
    r_traj,
    done_traj,
) -> None:
    # 만약 on-policy trajectory가 비어있다면 업데이트하지 않고 반환
    if len(s_traj) == 0:
        return

    s_traj_t = torch.as_tensor(np.array(s_traj), dtype=torch.float32, device=DEVICE)
    a_traj_t = torch.as_tensor(a_traj, dtype=torch.int64, device=DEVICE)
    r_traj_t = torch.as_tensor(r_traj, dtype=torch.float32, device=DEVICE)
    done_traj_t = torch.as_tensor(done_traj, dtype=torch.float32, device=DEVICE)

    # 마지막 상태는 부트스트랩하지 않고 0으로 단순 처리
    bootstrap = 0.0

    # 뒤에서부터 n-step return 계산
    returns = []
    R = bootstrap
    for r, d in zip(reversed(r_traj_t), reversed(done_traj_t)):
        R = r + GAMMA * R * (1.0 - d)
        returns.append(R)
    returns.reverse()
    returns_t = torch.stack(returns)  # (T,)

    pi, v = model(s_traj_t)  # pi: (T,2), v: (T,)
    dist = Categorical(pi)
    log_pi_a = dist.log_prob(a_traj_t)  # (T,)

    advantage = returns_t - v

    value_loss = F.mse_loss(v, returns_t.detach())
    policy_loss = -(log_pi_a * advantage.detach()).mean()
    entropy_loss = dist.entropy().mean()

    loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def acer_off_policy_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
    batch_size: int = BATCH_SIZE,
) -> None:
    # Replay Buffer에서 샘플링한 데이터를 이용한 off-policy ACER 스타일 업데이트.
    # - Importance Sampling 비율 w = π(a|s) / μ(a|s)
    # - w를 IS_CLIP으로 클리핑해서 분산을 억제
    # - A2C 스타일 advantage = r + γV(s') - V(s) 를 사용 (simple version)
    # 만약 Replay Buffer의 크기가 batch_size보다 작다면 업데이트하지 않고 반환
    if len(replay_buffer) < batch_size:
        return

    # Replay Buffer에서 샘플링한 데이터를 이용한 off-policy ACER 스타일 업데이트.
    s, a, r, done, s_next, mu = replay_buffer.sample(batch_size)

    s_t = torch.as_tensor(s, dtype=torch.float32, device=DEVICE)
    a_t = torch.as_tensor(a, dtype=torch.int64, device=DEVICE)
    r_t = torch.as_tensor(r, dtype=torch.float32, device=DEVICE)
    done_t = torch.as_tensor(done, dtype=torch.float32, device=DEVICE)
    s_next_t = torch.as_tensor(s_next, dtype=torch.float32, device=DEVICE)
    mu_t = torch.as_tensor(mu, dtype=torch.float32, device=DEVICE)  # behavior prob

    pi, v = model(s_t)  # pi: (B,2), v: (B,)
    _, v_next = model(s_next_t)

    dist = Categorical(pi)
    log_pi_a = dist.log_prob(a_t)
    pi_a = torch.exp(log_pi_a)  # π(a|s)

    # 중요도 비율 w = π(a|s) / μ(a|s)
    w = (pi_a / (mu_t + 1e-8)).detach()
    w_clipped = torch.clamp(w, max=IS_CLIP)

    # 1-step TD target
    td_target = r_t + GAMMA * v_next * (1.0 - done_t)
    advantage = td_target.detach() - v

    # Off-policy policy loss (클리핑된 중요도 비율 적용)
    policy_loss = -(w_clipped * log_pi_a * advantage.detach()).mean()

    # 가치 함수 손실
    value_loss = F.mse_loss(v, td_target.detach())

    entropy = dist.entropy().mean()

    loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


### 학습 루프
def train_acer() -> ActorCritic:
    env = gym.make(ENV_NAME)
    model = ActorCritic().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(REPLAY_CAPACITY)

    scores = []

    for episode in range(1, MAX_EPISODES + 1):
        s, _ = env.reset()
        done = False
        ep_reward = 0.0

        # on-policy trajectory 저장용
        s_traj, a_traj, r_traj, done_traj = [], [], [], []

        while not done:
            # 현재 정책에서 행동 샘플
            a, pi_probs, _ = model.act(s)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            ep_reward += r

            # on-policy trajectory에 저장
            s_traj.append(s)
            a_traj.append(a)
            r_traj.append(r / 100.0)  # 보상 스케일링
            done_traj.append(float(done))

            # Replay Buffer에 off-policy용 데이터 저장
            behavior_prob = float(pi_probs[a])  # μ(a|s)
            replay_buffer.push(s, a, r / 100.0, float(done), s_next, behavior_prob)

            s = s_next

        # 한 에피소드 끝나면 on-policy 업데이트 한 번
        acer_on_policy_update(model, optimizer, s_traj, a_traj, r_traj, done_traj)

        # Replay Buffer에서 여러 번 off-policy 업데이트
        if len(replay_buffer) >= REPLAY_START_SIZE:
            for _ in range(4):
                acer_off_policy_update(model, optimizer, replay_buffer, BATCH_SIZE)

        scores.append(ep_reward)

        if episode % 10 == 0:
            avg_score = sum(scores[-10:]) / 10.0
            print(
                f"[Episode {episode:4d}] "
                f"avg score (last 10) = {avg_score:.1f}, "
                f"buffer size = {len(replay_buffer)}"
            )

    env.close()
    return model


if __name__ == "__main__":
    train_acer()

