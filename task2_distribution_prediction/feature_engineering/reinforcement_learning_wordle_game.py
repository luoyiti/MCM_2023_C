import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from wordfreq import top_n_list

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None


WORD_LEN = 5
MAX_STEPS = 6

GRAY = 0
YELLOW = 1
GREEN = 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)


def _char2idx(c: str) -> int:
    return ord(c) - 97


def build_guess_vocab_from_wordfreq(top_n: int = 500000, limit: int = 20000) -> List[str]:
    top_words = top_n_list('en', int(top_n))
    words_5 = [w for w in top_words if len(w) == WORD_LEN if "'" not in w]
    out: List[str] = []
    for w in words_5:
        wl = w.lower()
        if wl.isalpha() and wl.isascii():
            out.append(wl)
        if len(out) >= int(limit):
            break
    return out


def build_vocab(
    answer_csv_path: str,
    top_n: int = 500000,
    limit: int = 20000,
) -> Tuple[List[str], List[str], Dict[str, int], Dict[int, str], List[int]]:
    allowed = build_guess_vocab_from_wordfreq(top_n=top_n, limit=limit)

    df = pd.read_csv(answer_csv_path, low_memory=False)
    answers = [str(w).strip().lower() for w in df['word'].tolist()]
    answers = [w for w in answers if len(w) == WORD_LEN and w.isalpha()]

    all_words: List[str] = []
    seen = set()
    for w in allowed + answers:
        if w in seen:
            continue
        seen.add(w)
        all_words.append(w)

    word2id = {w: i for i, w in enumerate(all_words)}
    id2word = {i: w for w, i in word2id.items()}
    answer_ids = [word2id[w] for w in answers if w in word2id]
    return all_words, answers, word2id, id2word, answer_ids


def precompute_word_arrays(all_words: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(all_words)
    letters = np.zeros((n, WORD_LEN), dtype=np.int8)
    masks = np.zeros((n,), dtype=np.int32)

    for i, w in enumerate(all_words):
        m = 0
        for j, ch in enumerate(w):
            idx = _char2idx(ch)
            if idx < 0 or idx >= 26:
                raise ValueError(f"invalid character {ch!r} in word {w!r}")
            letters[i, j] = idx
            m |= 1 << idx
        masks[i] = m

    return letters, masks


def compute_feedback(guess_word: str, answer_word: str) -> np.ndarray:
    feedback = np.full((WORD_LEN,), GRAY, dtype=np.int8)
    used = np.zeros((WORD_LEN,), dtype=bool)

    for i in range(WORD_LEN):
        if guess_word[i] == answer_word[i]:
            feedback[i] = GREEN
            used[i] = True

    for i in range(WORD_LEN):
        if feedback[i] == GREEN:
            continue
        g = guess_word[i]
        found = False
        for j in range(WORD_LEN):
            if (not used[j]) and (answer_word[j] == g):
                used[j] = True
                found = True
                break
        feedback[i] = YELLOW if found else GRAY

    return feedback


@dataclass
class WordleState:
    green_pos: np.ndarray
    must_have_mask: int
    not_allowed_mask: int
    step: int
    candidate_mask: np.ndarray


class WordleEnv:
    def __init__(
        self,
        id2word: Dict[int, str],
        answer_ids: List[int],
        word_letters: np.ndarray,
        word_letter_masks: np.ndarray,
        max_steps: int = MAX_STEPS,
    ):
        self.id2word = id2word
        self.answer_ids = np.array(answer_ids, dtype=np.int32)
        self.word_letters = word_letters
        self.word_letter_masks = word_letter_masks
        self.max_steps = int(max_steps)

        self.num_actions = int(word_letters.shape[0])
        self._base_candidate_mask = np.zeros((self.num_actions,), dtype=bool)
        self._base_candidate_mask[self.answer_ids] = True

        self.answer_id: Optional[int] = None
        self.state: Optional[WordleState] = None
        self.done = False

    def reset(self, answer_id: Optional[int] = None) -> WordleState:
        if answer_id is None:
            self.answer_id = int(np.random.choice(self.answer_ids))
        else:
            self.answer_id = int(answer_id)

        self.state = WordleState(
            green_pos=np.full((WORD_LEN,), -1, dtype=np.int8),
            must_have_mask=0,
            not_allowed_mask=0,
            step=0,
            candidate_mask=self._base_candidate_mask.copy(),
        )
        self.done = False
        return self.state

    def _update_state(self, guess_id: int, feedback: np.ndarray) -> WordleState:
        if self.state is None:
            raise RuntimeError('env not reset')
        s = self.state
        guess_word = self.id2word[int(guess_id)]

        for i in range(WORD_LEN):
            if feedback[i] == GREEN:
                s.green_pos[i] = _char2idx(guess_word[i])

        for i in range(WORD_LEN):
            if feedback[i] == YELLOW:
                s.must_have_mask |= 1 << _char2idx(guess_word[i])

        gy_letters = {guess_word[i] for i in range(WORD_LEN) if feedback[i] in (GREEN, YELLOW)}
        for i in range(WORD_LEN):
            if feedback[i] == GRAY:
                c = guess_word[i]
                if c not in gy_letters:
                    s.not_allowed_mask |= 1 << _char2idx(c)

        s.step += 1

        cand = s.candidate_mask
        if 0 <= guess_id < self.num_actions:
            cand[guess_id] = False

        for pos in range(WORD_LEN):
            gp = int(s.green_pos[pos])
            if gp != -1:
                cand &= self.word_letters[:, pos] == gp

        if s.not_allowed_mask:
            cand &= (self.word_letter_masks & s.not_allowed_mask) == 0

        if s.must_have_mask:
            cand &= (self.word_letter_masks & s.must_have_mask) == s.must_have_mask

        if not cand.any():
            cand = self._base_candidate_mask.copy()
            if 0 <= guess_id < self.num_actions:
                cand[guess_id] = False

        s.candidate_mask = cand
        return s

    def step(self, action_id: int):
        if self.done or self.state is None or self.answer_id is None:
            raise RuntimeError('episode already done or env not reset')

        guess_id = int(action_id)
        guess_word = self.id2word[guess_id]
        answer_word = self.id2word[int(self.answer_id)]

        feedback = compute_feedback(guess_word, answer_word)

        # 改进的奖励函数，增加信息增益奖励
        reward = -0.5  # 基础步数惩罚（降低以鼓励探索）
        done = False
        success = False
        
        # 计算反馈中的信息增益奖励
        green_count = int(np.sum(feedback == GREEN))
        yellow_count = int(np.sum(feedback == YELLOW))
        info_reward = green_count * 0.3 + yellow_count * 0.1  # 绿色和黄色反馈的奖励
        reward += info_reward
        
        if guess_word == answer_word:
            # 成功奖励，越早猜中奖励越高
            steps_used = self.state.step + 1
            reward += 10.0 + (MAX_STEPS - steps_used) * 1.0  # 早猜中额外奖励
            done = True
            success = True
        elif self.state.step + 1 >= self.max_steps:
            reward -= 5.0  # 失败惩罚
            done = True

        new_state = self._update_state(guess_id, feedback)
        self.done = done
        info = {'feedback': feedback, 'success': success}
        return new_state, reward, done, info


def encode_state_to_vector(state: WordleState, num_actions: int) -> np.ndarray:
    green_vec = np.zeros((WORD_LEN * 26,), dtype=np.float32)
    for i in range(WORD_LEN):
        gp = int(state.green_pos[i])
        if gp != -1:
            green_vec[i * 26 + gp] = 1.0

    must_vec = np.array([(state.must_have_mask >> i) & 1 for i in range(26)], dtype=np.float32)
    not_vec = np.array([(state.not_allowed_mask >> i) & 1 for i in range(26)], dtype=np.float32)

    step_vec = np.zeros((MAX_STEPS + 1,), dtype=np.float32)
    step_idx = int(state.step)
    if step_idx < 0:
        step_idx = 0
    if step_idx > MAX_STEPS:
        step_idx = MAX_STEPS
    step_vec[step_idx] = 1.0

    cand_size_norm = float(np.sum(state.candidate_mask)) / float(num_actions)
    return np.concatenate(
        [
            green_vec,
            must_vec,
            not_vec,
            step_vec,
            np.array([cand_size_norm], dtype=np.float32),
        ],
        axis=0,
    )


STATE_DIM = WORD_LEN * 26 + 26 + 26 + (MAX_STEPS + 1) + 1


@dataclass
class Trajectory:
    state_vecs: np.ndarray
    action_masks: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray


def _build_mlp(input_dim: int, output_dim: int, hidden_sizes: List[int]) -> tf.keras.Model:
    layers: List[tf.keras.layers.Layer] = [tf.keras.layers.Input(shape=(input_dim,), dtype=tf.float32)]
    for h in hidden_sizes:
        layers.append(tf.keras.layers.Dense(h, activation='relu'))
    layers.append(tf.keras.layers.Dense(output_dim, activation=None))
    return tf.keras.Sequential(layers)


def _apply_action_mask(logits: tf.Tensor, action_mask: tf.Tensor) -> tf.Tensor:
    neg_inf = tf.constant(-1e9, dtype=logits.dtype)
    return tf.where(action_mask, logits, neg_inf)


def compute_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    g = 0.0
    out = np.zeros_like(rewards, dtype=np.float32)
    for t in range(len(rewards) - 1, -1, -1):
        g = float(rewards[t]) + float(gamma) * g
        out[t] = g
    return out


class A2CAgent:
    def __init__(
        self,
        num_actions: int,
        state_dim: int = STATE_DIM,
        embed_dim: int = 128,
        hidden_sizes: Optional[List[int]] = None,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 1.0,
    ):
        self.num_actions = int(num_actions)
        self.state_dim = int(state_dim)
        self.embed_dim = int(embed_dim)
        self.gamma = float(gamma)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)

        if hidden_sizes is None:
            hidden_sizes = [256, 128]

        self.actor = _build_mlp(self.state_dim, self.embed_dim, hidden_sizes)
        self.critic = _build_mlp(self.state_dim, 1, hidden_sizes)

        init = tf.keras.initializers.GlorotUniform()
        self.action_embeddings = tf.Variable(
            init(shape=(self.num_actions, self.embed_dim), dtype=tf.float32),
            trainable=True,
            name='action_embeddings',
        )

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=float(actor_lr))
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=float(critic_lr))

    def policy_logits(self, state_vecs: tf.Tensor, action_masks: tf.Tensor) -> tf.Tensor:
        h = self.actor(state_vecs)
        logits = tf.matmul(h, self.action_embeddings, transpose_b=True)
        return _apply_action_mask(logits, action_masks)

    def value(self, state_vecs: tf.Tensor) -> tf.Tensor:
        v = self.critic(state_vecs)
        return tf.squeeze(v, axis=-1)

    def sample_action(self, state_vec: np.ndarray, action_mask: np.ndarray, epsilon: float = 0.0) -> int:
        if not action_mask.any():
            action_mask = np.ones((self.num_actions,), dtype=bool)

        if epsilon > 0.0 and random.random() < float(epsilon):
            idx = np.flatnonzero(action_mask)
            return int(np.random.choice(idx))

        s = tf.convert_to_tensor(state_vec.reshape(1, -1), dtype=tf.float32)
        m = tf.convert_to_tensor(action_mask.reshape(1, -1), dtype=tf.bool)
        logits = self.policy_logits(s, m)
        a = tf.random.categorical(logits, 1)[0, 0]
        return int(a.numpy())

    def greedy_action(self, state_vec: np.ndarray, action_mask: np.ndarray) -> int:
        if not action_mask.any():
            action_mask = np.ones((self.num_actions,), dtype=bool)
        s = tf.convert_to_tensor(state_vec.reshape(1, -1), dtype=tf.float32)
        m = tf.convert_to_tensor(action_mask.reshape(1, -1), dtype=tf.bool)
        logits = self.policy_logits(s, m)
        a = tf.argmax(logits[0], axis=-1)
        return int(a.numpy())

    def update(self, traj: Trajectory) -> Dict[str, float]:
        returns_np = compute_returns(traj.rewards, self.gamma)

        states = tf.convert_to_tensor(traj.state_vecs, dtype=tf.float32)
        masks = tf.convert_to_tensor(traj.action_masks, dtype=tf.bool)
        actions = tf.convert_to_tensor(traj.actions, dtype=tf.int32)
        returns = tf.convert_to_tensor(returns_np, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            values = self.value(states)
            logits = self.policy_logits(states, masks)
            log_probs_all = tf.nn.log_softmax(logits, axis=-1)
            probs_all = tf.nn.softmax(logits, axis=-1)

            idx = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
            selected_log_probs = tf.gather_nd(log_probs_all, idx)

            advantages = tf.stop_gradient(returns - values)
            actor_loss = -tf.reduce_mean(selected_log_probs * advantages)
            critic_loss = tf.reduce_mean(tf.square(values - returns))
            entropy = -tf.reduce_mean(tf.reduce_sum(probs_all * log_probs_all, axis=-1))
            actor_loss_total = actor_loss - self.entropy_coef * entropy

        actor_vars = list(self.actor.trainable_variables) + [self.action_embeddings]
        critic_vars = list(self.critic.trainable_variables)

        actor_grads = tape.gradient(actor_loss_total, actor_vars)
        critic_grads = tape.gradient(critic_loss, critic_vars)
        del tape

        actor_pairs = [(g, v) for g, v in zip(actor_grads, actor_vars) if g is not None]
        critic_pairs = [(g, v) for g, v in zip(critic_grads, critic_vars) if g is not None]

        if actor_pairs:
            g_list, v_list = zip(*actor_pairs)
            g_list, _ = tf.clip_by_global_norm(list(g_list), self.max_grad_norm)
            self.actor_optimizer.apply_gradients(zip(g_list, v_list))

        if critic_pairs:
            g_list, v_list = zip(*critic_pairs)
            g_list, _ = tf.clip_by_global_norm(list(g_list), self.max_grad_norm)
            self.critic_optimizer.apply_gradients(zip(g_list, v_list))

        return {
            'actor_loss': float(actor_loss.numpy()),
            'critic_loss': float(critic_loss.numpy()),
            'entropy': float(entropy.numpy()),
        }


def run_episode(env: WordleEnv, agent: A2CAgent, epsilon: float = 0.0) -> Tuple[Trajectory, bool]:
    state = env.reset()

    state_vecs: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    actions: List[int] = []
    rewards: List[float] = []
    done = False
    success = False

    while not done:
        s_vec = encode_state_to_vector(state, env.num_actions)
        a_mask = state.candidate_mask.astype(bool, copy=True)
        a = agent.sample_action(s_vec, a_mask, epsilon=epsilon)

        next_state, reward, done, info = env.step(a)
        success = bool(info.get('success', False))

        state_vecs.append(s_vec)
        masks.append(a_mask)
        actions.append(a)
        rewards.append(float(reward))

        state = next_state

    traj = Trajectory(
        state_vecs=np.stack(state_vecs, axis=0),
        action_masks=np.stack(masks, axis=0),
        actions=np.asarray(actions, dtype=np.int32),
        rewards=np.asarray(rewards, dtype=np.float32),
    )
    return traj, success


def train_agent(
    env: WordleEnv,
    agent: A2CAgent,
    num_episodes: int,
    epsilon_start: float = 0.1,
    epsilon_end: float = 0.0,
    epsilon_decay: float = 0.99995,
    log_every: int = 200,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 2000,
) -> None:
    eps = float(epsilon_start)
    win_window: List[int] = []
    steps_window: List[int] = []
    steps_distribution = np.zeros((MAX_STEPS + 1,), dtype=np.int32)  # 记录所有episode的猜测次数分布

    ckpt_manager = None
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt = tf.train.Checkpoint(
            actor=agent.actor,
            critic=agent.critic,
            action_embeddings=agent.action_embeddings,
            actor_opt=agent.actor_optimizer,
            critic_opt=agent.critic_optimizer,
        )
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

    for ep in range(1, int(num_episodes) + 1):
        traj, success = run_episode(env, agent, epsilon=eps)
        metrics = agent.update(traj)

        win_window.append(1 if success else 0)
        steps = len(traj.rewards)
        steps_window.append(steps)
        
        # 更新分布统计
        if success:
            idx = min(steps, MAX_STEPS) - 1
            if idx < 0:
                idx = 0
            steps_distribution[idx] += 1
        else:
            steps_distribution[MAX_STEPS] += 1
        
        if len(win_window) > 200:
            win_window.pop(0)
            steps_window.pop(0)

        eps = max(float(epsilon_end), eps * float(epsilon_decay))

        if log_every and ep % int(log_every) == 0:
            win_rate = float(np.mean(win_window)) if win_window else 0.0
            avg_steps = float(np.mean(steps_window)) if steps_window else 0.0
            
            # 计算并显示分布
            total_eps = sum(steps_distribution)
            if total_eps > 0:
                dist_percent = steps_distribution / total_eps * 100
                dist_str = " | ".join([f"{i+1}次:{dist_percent[i]:.1f}%" if i < MAX_STEPS else f"失败:{dist_percent[i]:.1f}%" 
                                       for i in range(MAX_STEPS + 1)])
            else:
                dist_str = "无数据"
            
            print(
                f"Episode {ep}/{num_episodes} | win_rate={win_rate:.3f} | avg_steps={avg_steps:.2f} | "
                f"分布[{dist_str}] | "
                f"actor_loss={metrics['actor_loss']:.4f} | critic_loss={metrics['critic_loss']:.4f} | "
                f"entropy={metrics['entropy']:.4f} | eps={eps:.4f}"
            )

        if ckpt_manager is not None and checkpoint_every and ep % int(checkpoint_every) == 0:
            ckpt_manager.save(checkpoint_number=ep)

    if ckpt_manager is not None:
        ckpt_manager.save(checkpoint_number=int(num_episodes))
    
    # 训练结束后打印最终分布
    print("\n=== 训练完成后的猜测次数分布 ===")
    total_eps = sum(steps_distribution)
    if total_eps > 0:
        dist_percent = steps_distribution / total_eps * 100
        print(f"总训练轮数: {total_eps}")
        for i in range(MAX_STEPS + 1):
            label = f"{i+1}次猜对" if i < MAX_STEPS else "失败"
            print(f"  {label}: {steps_distribution[i]} 次 ({dist_percent[i]:.1f}%)")


def load_agent_from_checkpoint(agent: A2CAgent, checkpoint_dir: str) -> None:
    ckpt = tf.train.Checkpoint(
        actor=agent.actor,
        critic=agent.critic,
        action_embeddings=agent.action_embeddings,
        actor_opt=agent.actor_optimizer,
        critic_opt=agent.critic_optimizer,
    )
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(f'No checkpoint found in {checkpoint_dir}')
    ckpt.restore(latest).expect_partial()


def evaluate_word_difficulty(
    env: WordleEnv,
    agent: A2CAgent,
    answer_id: int,
    num_runs: int = 500,
    eval_epsilon: float = 0.1,
    use_sampling: bool = True,
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """
    评估单词难度，支持随机性策略。
    
    Args:
        eval_epsilon: 评估时的探索率（ε-贪婪策略）
        use_sampling: 是否使用采样策略（True）还是贪婪策略（False）
    """
    steps_count = np.zeros((MAX_STEPS + 1,), dtype=np.int32)
    success_count = 0
    
    # 记录每次猜测的具体结果
    individual_results = []

    for run_idx in range(int(num_runs)):
        state = env.reset(answer_id=int(answer_id))
        done = False
        step = 0
        info = {'success': False}

        while not done:
            s_vec = encode_state_to_vector(state, env.num_actions)
            a_mask = state.candidate_mask.astype(bool, copy=False)
            
            # 使用采样策略而非贪婪策略，引入随机性
            if use_sampling:
                action = agent.sample_action(s_vec, a_mask, epsilon=eval_epsilon)
            else:
                action = agent.greedy_action(s_vec, a_mask)
            
            state, _, done, info = env.step(action)
            step += 1

        if bool(info.get('success', False)):
            success_count += 1
            idx = min(step, MAX_STEPS) - 1
            if idx < 0:
                idx = 0
            steps_count[idx] += 1
            individual_results.append(step)
        else:
            steps_count[MAX_STEPS] += 1
            individual_results.append(-1)  # -1 表示失败

    p = steps_count.astype(np.float32) / float(num_runs)
    p_success = float(success_count) / float(num_runs)
    e_t = float(np.sum(np.arange(1, MAX_STEPS + 2, dtype=np.float32) * p))
    return p, p_success, e_t, np.array(individual_results)


def evaluate_all_words(
    env: WordleEnv,
    agent: A2CAgent,
    answer_ids: List[int],
    id2word: Dict[int, str],
    num_runs: int = 500,
    eval_epsilon: float = 0.1,
    use_sampling: bool = True,
) -> pd.DataFrame:
    """
    评估所有单词的难度分布。
    
    Args:
        eval_epsilon: 评估时的探索率
        use_sampling: 是否使用采样策略
    """
    rows = []
    for aid in answer_ids:
        p, p_success, e_t, individual_results = evaluate_word_difficulty(
            env, agent, aid, num_runs=num_runs,
            eval_epsilon=eval_epsilon, use_sampling=use_sampling
        )
        
        # 统计真实分布
        steps_1 = np.sum(individual_results == 1)
        steps_2 = np.sum(individual_results == 2)
        steps_3 = np.sum(individual_results == 3)
        steps_4 = np.sum(individual_results == 4)
        steps_5 = np.sum(individual_results == 5)
        steps_6 = np.sum(individual_results == 6)
        failures = np.sum(individual_results == -1)
        
        # 计算归一化的真实分布（概率）
        real_p1 = float(steps_1 / num_runs)
        real_p2 = float(steps_2 / num_runs)
        real_p3 = float(steps_3 / num_runs)
        real_p4 = float(steps_4 / num_runs)
        real_p5 = float(steps_5 / num_runs)
        real_p6 = float(steps_6 / num_runs)
        real_p_fail = float(failures / num_runs)
        
        # 计算真实分布的平均期望值（失败当作第7步）
        real_expected_steps = float(
            1 * real_p1 + 2 * real_p2 + 3 * real_p3 + 4 * real_p4 + 
            5 * real_p5 + 6 * real_p6 + 7 * real_p_fail
        )
        
        row = {
            'word': id2word[int(aid)],
            'rl_success': p_success,
            'rl_expected_steps': e_t,
            'rl_p1': float(p[0]),
            'rl_p2': float(p[1]),
            'rl_p3': float(p[2]),
            'rl_p4': float(p[3]),
            'rl_p5': float(p[4]),
            'rl_p6': float(p[5]),
            'rl_fail': float(p[6]),
            # 添加真实次数统计
            'count_1': int(steps_1),
            'count_2': int(steps_2),
            'count_3': int(steps_3),
            'count_4': int(steps_4),
            'count_5': int(steps_5),
            'count_6': int(steps_6),
            'count_fail': int(failures),
            # 添加归一化的真实分布（概率）
            'real_p1': real_p1,
            'real_p2': real_p2,
            'real_p3': real_p3,
            'real_p4': real_p4,
            'real_p5': real_p5,
            'real_p6': real_p6,
            'real_p_fail': real_p_fail,
            # 添加真实分布的平均期望值
            'real_expected_steps': real_expected_steps,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    
    # 打印真实分布统计信息
    print("\n=== 真实猜测次数分布统计 ===")
    print(f"每个单词测试 {num_runs} 次")
    print("\n平均分布（所有单词）:")
    avg_dist = df[['count_1', 'count_2', 'count_3', 'count_4', 'count_5', 'count_6', 'count_fail']].mean()
    for i, col in enumerate(['count_1', 'count_2', 'count_3', 'count_4', 'count_5', 'count_6', 'count_fail']):
        guess_num = i + 1 if i < 6 else '失败'
        print(f"  {guess_num}次: {avg_dist[col]:.1f} 次 ({avg_dist[col]/num_runs*100:.1f}%)")
    
    # 打印归一化的真实分布（概率）
    print("\n归一化的真实分布概率（所有单词平均）:")
    avg_real_dist = df[['real_p1', 'real_p2', 'real_p3', 'real_p4', 'real_p5', 'real_p6', 'real_p_fail']].mean()
    for i, col in enumerate(['real_p1', 'real_p2', 'real_p3', 'real_p4', 'real_p5', 'real_p6', 'real_p_fail']):
        guess_num = i + 1 if i < 6 else '失败'
        print(f"  {guess_num}次: {avg_real_dist[col]:.3f}")
    
    # 打印期望值对比
    print("\n期望值对比:")
    avg_rl_expected = df['rl_expected_steps'].mean()
    avg_real_expected = df['real_expected_steps'].mean()
    print(f"  RL模型期望步数: {avg_rl_expected:.2f}")
    print(f"  真实分布期望步数: {avg_real_expected:.2f}")
    
    print("\n最难的10个单词（真实期望步数最多）:")
    hardest = df.nlargest(10, 'real_expected_steps')[['word', 'real_expected_steps', 'count_1', 'count_2', 'count_3', 'count_4', 'count_5', 'count_6', 'count_fail']]
    print(hardest.to_string(index=False))
    
    print("\n最简单的10个单词（真实期望步数最少）:")
    easiest = df.nsmallest(10, 'real_expected_steps')[['word', 'real_expected_steps', 'count_1', 'count_2', 'count_3', 'count_4', 'count_5', 'count_6', 'count_fail']]
    print(easiest.to_string(index=False))
    
    # 显示前20个单词的详细真实分布
    print("\n=== 前20个单词的真实分布 ===")
    for i, (_, row) in enumerate(df.head(20).iterrows()):
        word = row['word']
        counts = [row['count_1'], row['count_2'], row['count_3'], row['count_4'], row['count_5'], row['count_6'], row['count_fail']]
        max_count_idx = np.argmax(counts)
        if max_count_idx < 6:
            most_likely = f"{max_count_idx + 1}次"
        else:
            most_likely = "失败"
        print(f"{word}: 1次:{counts[0]} 2次:{counts[1]} 3次:{counts[2]} 4次:{counts[3]} 5次:{counts[4]} 6次:{counts[5]} 失败:{counts[6]} (最可能:{most_likely})")
    
    return df


def merge_rl_metrics_into_csv(
    base_csv_path: str,
    rl_df: pd.DataFrame,
    out_csv_path: str,
) -> None:
    base = pd.read_csv(base_csv_path, low_memory=False)
    merged = base.merge(rl_df, on='word', how='left')
    merged.to_csv(out_csv_path, index=False)


def main() -> None:
    base_dir = os.path.dirname(__file__)
    default_answers = os.path.join(base_dir, 'data', 'mcm_processed_data.csv')
    default_ckpt = os.path.join(base_dir, 'models', 'wordle_a2c_ckpt')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'evaluate', 'train_evaluate'], default='train')
    parser.add_argument('--answer_csv_path', default=default_answers)
    parser.add_argument('--guess_top_n', type=int, default=500000)
    parser.add_argument('--guess_limit', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--episodes', type=int, default=20000)
    parser.add_argument('--epsilon_start', type=float, default=0.1)
    parser.add_argument('--epsilon_end', type=float, default=0.0)
    parser.add_argument('--epsilon_decay', type=float, default=0.9999)
    parser.add_argument('--log_every', type=int, default=200)

    parser.add_argument('--checkpoint_dir', default=default_ckpt)
    parser.add_argument('--checkpoint_every', type=int, default=2000)

    parser.add_argument('--eval_runs', type=int, default=200)
    parser.add_argument('--eval_limit', type=int, default=0)
    parser.add_argument('--eval_epsilon', type=float, default=0.1, help='评估时的探索率（ε-贪婪策略）')
    parser.add_argument('--eval_greedy', action='store_true', help='评估时使用贪婪策略（默认使用采样策略）')
    parser.add_argument('--eval_out_csv', default=os.path.join(base_dir, 'data', 'rl_word_difficulty.csv'))
    parser.add_argument('--merge_out_csv', default=os.path.join(base_dir, 'data', 'mcm_processed_data_with_rl.csv'))

    args = parser.parse_args()
    set_seed(int(args.seed))

    if tf is None:
        raise RuntimeError(
            "TensorFlow is not installed in the current Python environment. "
            "Please install tensorflow-macos (and optionally tensorflow-metal) or run this script in an environment with TensorFlow."
        )

    all_words, _, _, id2word, answer_ids = build_vocab(
        args.answer_csv_path,
        top_n=int(args.guess_top_n),
        limit=int(args.guess_limit),
    )
    word_letters, word_letter_masks = precompute_word_arrays(all_words)

    env = WordleEnv(
        id2word=id2word,
        answer_ids=answer_ids,
        word_letters=word_letters,
        word_letter_masks=word_letter_masks,
    )
    # 增加熵正则化系数，鼓励策略多样性
    agent = A2CAgent(
        num_actions=env.num_actions,
        entropy_coef=0.01,  # 熵正则化，鼓励探索
        hidden_sizes=[512, 256, 128],  # 更复杂的网络结构
    )

    if args.mode in ('train', 'train_evaluate'):
        train_agent(
            env,
            agent,
            num_episodes=int(args.episodes),
            epsilon_start=float(args.epsilon_start),
            epsilon_end=float(args.epsilon_end),
            epsilon_decay=float(args.epsilon_decay),
            log_every=int(args.log_every),
            checkpoint_dir=str(args.checkpoint_dir) if args.checkpoint_dir else None,
            checkpoint_every=int(args.checkpoint_every),
        )
    else:
        load_agent_from_checkpoint(agent, str(args.checkpoint_dir))

    if args.mode in ('evaluate', 'train_evaluate'):
        eval_answer_ids = answer_ids
        if int(args.eval_limit) > 0:
            eval_answer_ids = answer_ids[: int(args.eval_limit)]
        rl_df = evaluate_all_words(
            env,
            agent,
            answer_ids=eval_answer_ids,
            id2word=id2word,
            num_runs=int(args.eval_runs),
            eval_epsilon=float(args.eval_epsilon),
            use_sampling=not args.eval_greedy,
        )
        rl_df.to_csv(str(args.eval_out_csv), index=False)
        merge_rl_metrics_into_csv(str(args.answer_csv_path), rl_df, str(args.merge_out_csv))


if __name__ == '__main__':
    main()
