# ACER CartPole – Actor-Critic with Experience Replay (PyTorch)

이 저장소는 **ACER(Actor-Critic with Experience Replay)** 아이디어를  
`CartPole-v1` 환경에 적용한 **학습용 구현 예제**입니다.

논문 ACER 전체(Trust Region, Q-리트레이스 등)를 100% 재현하기보다는,

- Actor-Critic + Experience Replay  
- Off-policy 보정을 위한 Importance Sampling  
- on-policy / off-policy 업데이트를 섞어서 학습  

이라는 **핵심 구조를 이해하기 쉽게 정리한 코드**를 담고 있습니다.

---

## 1. 환경 설정

```bash
pip install gymnasium[classic-control]
pip install torch
pip install numpy
```

- Python 3.9+ 권장
- GPU가 있으면 자동으로 `cuda`를 사용하고, 없으면 `cpu`에서 동작합니다.

---

## 2. 파일 구성

- `acer_cartpole.py`  
  - CartPole용 ACER 스타일 학습 코드  
  - 주요 구성:
    - `ActorCritic` 네트워크
    - `ReplayBuffer`
    - `acer_on_policy_update` – A2C 스타일 on-policy 업데이트
    - `acer_off_policy_update` – Replay + Importance Sampling 기반 off-policy 업데이트
    - `train_acer()` – CartPole 에피소드 반복 학습 루프
---

## 3. 실행 방법

프로젝트 루트(이 파일이 있는 디렉터리)에서:

```bash
python acer_cartpole.py
```

실행하면 콘솔에 10 에피소드 단위로 평균 점수와 Replay Buffer 크기가 출력됩니다.

예시 로그:

```text
[Episode   10] avg score (last 10) = 20.3, buffer size = 200
[Episode   20] avg score (last 10) = 35.1, buffer size = 400
[Episode   30] avg score (last 10) = 75.8, buffer size = 600
...
```

점수가 점점 올라가면, ACER 스타일 업데이트가 CartPole을 점차 잘 해결하는 정책을 학습하고 있다는 뜻입니다.

---

## 4. 코드 구조 개요

### 4.1 ActorCritic 네트워크

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.pi_head = nn.Linear(hidden_dim, action_dim)
        self.v_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.pi_head(x)
        v = self.v_head(x)
        pi = F.softmax(logits, dim=-1)
        return pi, v.squeeze(-1)

      def act(self, s: np.ndarray):
        with torch.no_grad():
            s_t = torch.as_tensor(s, dtype=torch.float32, device=DEVICE)
            pi, v = self.forward(s_t)
            dist = Categorical(pi)
            a = dist.sample()
            return a.item(), pi.cpu().numpy(), float(v.item())
```

- CartPole 상태(4차원)를 입력으로 받아:
  - 정책 \(\pi(a|s)\) (두 행동에 대한 확률)
  - 상태 가치 \(V(s)\)
- `act()`는 환경 상호작용 시 사용되며,  
  **behavior 정책 확률 μ(a|s)** 를 Replay Buffer에 함께 저장합니다.

---

### 4.2 Replay Buffer & Importance Sampling

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, done, s_next, behavior_prob):
        self.buffer.append((s, a, r, done, s_next, behavior_prob))
```

- `(s, a, r, done, s_next, behavior_prob)` 저장
  - `behavior_prob` = 데이터를 수집할 때 사용했던 정책 확률 μ(a|s)
- off-policy 업데이트 시:
  - 현재 정책 π(a|s)를 다시 계산하고
  - **중요도 비율** \(w = \frac{\pi(a|s)}{\mu(a|s)}\)를 사용
  - 너무 큰 w는 `torch.clamp(w, max=IS_CLIP)`으로 클리핑

---

### 4.3 on-policy vs off-policy 업데이트

- `acer_on_policy_update(...)`  
  - 에피소드마다 1회 호출  
  - 방금 수집한 trajectory로 A2C 스타일 n-step 업데이트

- `acer_off_policy_update(...)`  
  - Replay Buffer에서 여러 번 샘플링  
  - Importance Sampling을 적용한 off-policy 업데이트

학습 루프에서는 대략 다음 형태로 사용합니다.

```python
# 1) 에피소드마다 on-policy 업데이트
acer_on_policy_update(model, optimizer, s_traj, a_traj, r_traj, done_traj)

# 2) Replay Buffer가 충분히 채워지면 off-policy 업데이트 여러 번
if len(replay_buffer) >= REPLAY_START_SIZE:
    for _ in range(4):
        acer_off_policy_update(model, optimizer, replay_buffer, BATCH_SIZE)
```

---

## 5. 이 구현이 “진짜 ACER”와 다른 점

- 논문 **“Sample Efficient Actor-Critic with Experience Replay (ACER)”** 전체 구현이 아니라,
  - Actor-Critic + Replay Buffer + Importance Sampling 이라는 **핵심 아이디어**만 담은 **축약 버전**입니다.
- 생략/단순화한 부분:
  - Trust Region / Bias Correction Term
  - Q-리트레이스(Q-Retrace) 및 Truncated IS의 세부 수식 등
  
이 저장소는 ACER 알고리즘의 구조를 학습하고 실험해 볼 수 있는 미니멀 예제로,  
알고리즘의 직관을 파악하는 데 초점을 둔 참고용 코드입니다.

---

## 6. 참고

- **원 논문**:  
  *Sample Efficient Actor-Critic with Experience Replay*  
  (Wang et al., 2016) – ACER 제안 논문

