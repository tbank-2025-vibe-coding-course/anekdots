#!/usr/bin/env python3
"""
markov_generator.py

Рабочий Markov-анекдот-генератор (всё в одном файле).
Требует стандартные библиотеки Python.

Пример:
    python markov_generator.py --dataset anekdots.csv --N 5 --threshold 5

После сборки моделей программа перейдет в интерактивный режим:
вводите ключевые слова через запятую (например: программист, начальник),
нажмите Enter — получите сгенерированный анекдот.
Выход: Ctrl+C.
"""

import argparse
import csv
import random
import re
import sys
import math
from collections import defaultdict, Counter, deque

# -------------------------
# Токенизация (только слова, без пунктуации)
# -------------------------
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

def tokenize(text):
    """Возвращает список слов (нижний регистр, 'ё'->'е'), без пунктуации."""
    if not text:
        return []
    s = text.lower().replace("ё", "е").replace('-\n', '').replace('й', 'й')
    # извлекаем "слова" - буквы/цифры/подчёрки; это убирает кавычки, скобки, знаки и т.п.
    return WORD_RE.findall(s)

def detokenize(tokens):
    """Простая склейка слов в предложение. Поднимаем первую букву и ставим точку, если нужно."""
    if not tokens:
        return ""
    s = " ".join(tokens).strip()
    if not s:
        return ""
    # Убираем возможные повторяющиеся пробелы (на всякий случай)
    s = re.sub(r"\s+", " ", s)
    # Capitalize first letter (unicode aware)
    first = s[0].upper()
    s = first + s[1:]
    # Ensure final punctuation: поставим точку, если нет '.', '!', '?'
    if s[-1] not in ".!?":
        s = s + "."
    return s

def longest_common_prefix_len(a, b):
    """Длина общего префикса двух строк."""
    la = len(a)
    lb = len(b)
    L = min(la, lb)
    i = 0
    while i < L and a[i] == b[i]:
        i += 1
    return i

def lcp_ratio(a, b):
    """Отношение длины LCP к большей длине слова, как вы просили."""
    if not a or not b:
        return 0.0
    return longest_common_prefix_len(a, b) / max(len(a), len(b))

def weighted_choice(items, weights):
    """Выбрать элемент из items по весам (weights). items: list, weights: list чисел."""
    # defensive: empty
    if not items:
        return None
    tot = 0.0
    cum = []
    for w in weights:
        tot += w
        cum.append(tot)
    if tot <= 0:
        # fallback to uniform
        return random.choice(items)
    r = random.random() * tot
    # binary search
    lo, hi = 0, len(cum) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if r < cum[mid]:
            hi = mid
        else:
            lo = mid + 1
    return items[lo]

# ---------------------------
# Model building
# ---------------------------

class MarkovModels:
    def __init__(self, N=5):
        self.N = N
        # models[m]: dict state_tuple -> Counter(next_token->count)
        self.models = {m: defaultdict(Counter) for m in range(1, N+1)}
        self.state_freqs = {m: defaultdict(int) for m in range(1, N+1)}
        self.unigram_counts = Counter()
        # For start-state selection: mapping token -> set(states)
        self.start_index = defaultdict(list)  # token -> list of states (each state is tuple length N)
        self.total_unigrams = 0

    def add_sequence(self, tokens):
        """Добавить токены (один анекдот) в модели. Добавим BOS/EOS контексты."""
        if tokens is None:
            return
        # Insert BOS markers to allow starting states (N times)
        BOS = ["<BOS>"] * self.N
        EOS = ["<EOS>"]
        seq = BOS + tokens + EOS
        L = len(seq)
        # unigrams
        for tok in seq:
            self.unigram_counts[tok] += 1
            self.total_unigrams += 1
        for m in range(1, self.N + 1):
            for i in range(0, L - m):
                state = tuple(seq[i: i + m])
                next_token = seq[i + m]
                self.models[m][state][next_token] += 1
                self.state_freqs[m][state] += 1
                # build start index only for states of length N that appear at beginning (i < N+1 maybe)
                # but we want all states length N for start selection; we'll index all N-grams by their tokens
                if m == self.N:
                    # index each token in the state for faster search of states containing a token
                    for tok in state:
                        self.start_index[tok].append(state)

    def build_from_texts(self, texts):
        """texts: iterable of strings (each is one joke)."""
        for text in texts:
            tokens = tokenize(text)
            self.add_sequence(tokens)

# ---------------------------
# Start-state selection (topic-based)
# ---------------------------

def score_state_by_prompt(state, prompt_norms, state_freq, lcp_threshold):
    """Простейшая эвристика: суммируем очки за совпадения LCP с prompt_norms.
    Более длинные совпадения дают больше очков; добавляем частотный бонус.
    Возвращаем score>0 если кандидат релевантен.
    """
    score = 0.0
    for pw in prompt_norms:
        for tok in state:
            ratio = lcp_ratio(tok, pw)
            if ratio > lcp_threshold:
                # качество совпадения: больше ratio -> больше score
                score += ratio * 2.0
    # frequency bonus (логарифмический) — чтобы не брать редкие старты
    score *= (1.0 + math.log1p(state_freq))
    return score

def choose_start_state(models_obj, prompt_words, N, top_k=50, lcp_threshold=0.4):
    """Выбрать стартовое состояние длины N.
    - Если prompt_words пусты — случайный частотный старт.
    - Иначе — найти кандидатов через индекс и LCP scoring.
    """
    if not prompt_words:
        # choose random frequent state (sample proportional to state_freq)
        states = list(models_obj.models[N].keys())
        if not states:
            return None
        # sample by freq
        weights = [models_obj.state_freqs[N].get(s, 0) for s in states]
        return weighted_choice(states, weights)

    prompt_norms = [pw.strip().lower().replace("ё", "е") for pw in prompt_words if pw.strip()]
    candidates = []
    seen = set()
    # use start_index: find states containing tokens that have potential LCP with prompt words
    for pw in prompt_norms:
        # we will inspect all tokens in start_index keys and compare via LCP ratio
        # to avoid iterating over all tokens we look up tokens that begin with pw[:3] etc.
        # but for simplicity (and given reasonable corpus size) we iterate over states found via exact token matches in index
        # gather potential states from tokens that are exact equals to pw or similar tokens
        for token, states_list in models_obj.start_index.items():
            # quick check: only consider tokens with same first char to reduce checks
            if not token: 
                continue
            # quick char check
            if token[0] != pw[0]:
                continue
            # compute lcp ratio
            if lcp_ratio(token, pw) >= lcp_threshold:
                for st in states_list:
                    if st not in seen:
                        seen.add(st)
                        freq = models_obj.state_freqs[N].get(st, 0)
                        score = score_state_by_prompt(st, prompt_norms, freq, lcp_threshold)
                        if score > 0:
                            candidates.append((st, score, freq))
    # If we have candidates, sample from them weighted by score*freq (top-K)
    if candidates:
        candidates.sort(key=lambda x: x[1] * (1.0 + math.log1p(x[2])), reverse=True)
        chosen_pool = candidates[:top_k]
        items = [c[0] for c in chosen_pool]
        weights = [c[1] * (1.0 + math.log1p(c[2])) for c in chosen_pool]
        return weighted_choice(items, weights)
    # fallback: random frequent state
    states = list(models_obj.models[N].keys())
    if not states:
        return None
    weights = [models_obj.state_freqs[N].get(s, 0) for s in states]
    return weighted_choice(states, weights)

# ---------------------------
# Generation core
# ---------------------------

def generate_one(models_obj, prompt_words,
                 N=5,
                 threshold=5,
                 T=1.0,
                 k_check=6,
                 L_recent=12,
                 alpha=2.0,
                 p_force=0.15,
                 max_length=120,
                 min_length=15,
                 lcp_threshold=0.4,
                 seed=None):
    """Сгенерировать один анекдот согласно спецификации."""
    if seed is not None:
        random.seed(seed)
    # prepare prompt norms (only lowercase replace ё->е)
    prompt_norms = [pw.strip().lower().replace("ё", "е") for pw in prompt_words if pw.strip()] if prompt_words else []

    start_state = choose_start_state(models_obj, prompt_norms, N, top_k=50, lcp_threshold=lcp_threshold)
    # if no state available, fallback to unigram start
    if start_state is None:
        # pick most frequent unigram as start
        if models_obj.unigram_counts:
            tok = weighted_choice(list(models_obj.unigram_counts.keys()),
                                  list(models_obj.unigram_counts.values()))
            start_state = tuple(["<BOS>"] * (N - 1) + [tok])
        else:
            start_state = tuple(["<BOS>"] * N)
    # build initial result: exclude leading <BOS> tokens
    result = [tok for tok in start_state if tok != "<BOS>"]
    current_state = tuple(start_state)  # length N initially
    m = len(current_state)
    steps = 0
    # ensure current_state always has length m; we'll manage slicing
    # We'll maintain a deque of last tokens for recent window
    recent = deque(result, maxlen=L_recent)
    while steps < max_length:
        # Backoff / shrink window until find state with freq >= threshold or m==0
        found_candidates = None
        cur_state = current_state[-m:] if m > 0 else tuple()
        while True:
            if m > 0 and cur_state in models_obj.models[m] and models_obj.state_freqs[m].get(cur_state, 0) >= threshold:
                found_candidates = models_obj.models[m][cur_state]
                break
            # if m>0 but state exists with smaller freq or not exists -> shrink
            if m > 0:
                # shrink window (remove leftmost)
                m -= 1
                cur_state = cur_state[1:] if len(cur_state) > 1 else tuple()
                if m == 0:
                    # will fall back to unigram
                    found_candidates = models_obj.unigram_counts
                    break
                continue
            else:
                # m == 0 -> unigram fallback
                found_candidates = models_obj.unigram_counts
                break

        # found_candidates is a Counter-like
        if not found_candidates:
            # dead-end: fallback to most frequent unigram or restart
            if models_obj.unigram_counts:
                found_candidates = models_obj.unigram_counts
            else:
                break

        # Prepare candidates list and counts
        tokens = list(found_candidates.keys())
        counts = [found_candidates[t] for t in tokens]

        # Steering check (every k_check steps)
        do_steer = False
        if prompt_norms:
            if (steps % k_check) == 0:
                # check recent window for presence of prompt
                recent_has = False
                for tok in recent:
                    for pw in prompt_norms:
                        if lcp_ratio(tok, pw) >= lcp_threshold:
                            recent_has = True
                            break
                    if recent_has:
                        break
                if not recent_has:
                    do_steer = True

        if do_steer:
            # find candidates that match prompt by LCP ratio and boost their counts
            boosted_counts = []
            matching_indices = []
            for i, tok in enumerate(tokens):
                tok_norm = tok.lower().replace("ё", "е")
                matched = any(lcp_ratio(tok_norm, pw) >= lcp_threshold for pw in prompt_norms)
                if matched:
                    matching_indices.append(i)
                    boosted_counts.append(counts[i] * (1.0 + alpha))
                else:
                    boosted_counts.append(counts[i])
            # with probability p_force choose one matching candidate directly (if any)
            if matching_indices and random.random() < p_force:
                chosen_tok = random.choice([tokens[i] for i in matching_indices])
                next_token = chosen_tok
            else:
                # temperature sampling with boosted_counts
                weights = [c ** (1.0 / max(1e-9, T)) for c in boosted_counts]
                next_token = weighted_choice(tokens, weights)
        else:
            # temperature sampling normally
            weights = [c ** (1.0 / max(1e-9, T)) for c in counts]
            next_token = weighted_choice(tokens, weights)

        if next_token is None:
            break

        # Append token
        if next_token == "<EOS>":
            # End of sequence token: if min_length satisfied -> stop, else continue with unigram choice
            if steps >= min_length:
                break
            else:
                # skip EOS and continue (choose another)
                # remove selected EOS from candidates temporarily
                # fallback: pick most frequent unigram instead
                if models_obj.unigram_counts:
                    next_token = weighted_choice(list(models_obj.unigram_counts.keys()),
                                                 list(models_obj.unigram_counts.values()))
                else:
                    break

        result.append(next_token)
        recent.append(next_token)

        # Update current_state: append token and keep last N tokens in tuple
        # But m might be < N currently; we maintain current_state as last N tokens (including <BOS> prefix)
        # Simpler: concatenate previous current_state (as tuple) with next_token, then slice last N
        # However we must ensure we preserve the relative alignment for window shrink logic: we will always compute cur_state = last m tokens when needed.
        # Build new full state (we won't keep older than N)
        prev_full = list(current_state)
        prev_full.append(next_token)
        if len(prev_full) > N:
            prev_full = prev_full[-N:]
        current_state = tuple(prev_full)
        # After appending, reset m to min(N, len(current_state)) for next step (optimistic)
        m = min(N, len(current_state))
        steps += 1

        # Stop if encountered end punctuation and min_length is satisfied
        if steps >= min_length and result:
            lasttok = result[-1]
            if lasttok in {".", "!", "?"}:
                break

    # Final postprocessing: remove leading <BOS> if any left in the front of result
    cleaned = [t for t in result if t != "<BOS>"]
    out = detokenize(cleaned)
    # Simple cleanup: strip spaces
    return out.strip()

# ---------------------------
# CSV loader
# ---------------------------

def load_texts_from_csv(path, text_column="text"):
    """Загрузить тексты из CSV, вернуть список строк (каждая строка - один анекдот).
    CSV может иметь заголовок; текстовое поле должно быть указано в text_column.
    """
    texts = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if text_column not in reader.fieldnames:
            # fallback: если нет header или другой формат, попробуем прочитать все строки в первую колонку
            f.seek(0)
            simple_reader = csv.reader(f)
            for row in simple_reader:
                if not row:
                    continue
                texts.append(row[0])
            return texts
        for row in reader:
            txt = row.get(text_column, "")
            if txt is None:
                txt = ""
            texts.append(txt)
    return texts

# ---------------------------
# CLI and main loop
# ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Markov joke generator (interactive).")
    ap.add_argument("--dataset", "-d", required=True, help="CSV dataset path (column 'text' with one joke per row).")
    ap.add_argument("--N", type=int, default=5, help="Maximum n-gram order (default 5).")
    ap.add_argument("--threshold", type=int, default=20, help="Min transitions for using a state without backoff (default 20).")
    ap.add_argument("--T", type=float, default=1.0, help="Temperature for sampling (default 1.0).")
    ap.add_argument("--k_check", type=int, default=6, help="Check steering every k tokens (default 6).")
    ap.add_argument("--L_recent", type=int, default=12, help="Length of recent window to check prompt presence (default 12).")
    ap.add_argument("--alpha", type=float, default=2.0, help="Steering boost multiplier (default 2.0).")
    ap.add_argument("--p_force", type=float, default=0.7, help="Probability to force-select a thematic candidate (default 0.7).")
    ap.add_argument("--lcp_threshold", type=float, default=0.75, help="LCP ratio threshold for considering words 'related' (default 0.75).")
    ap.add_argument("--max_length", type=int, default=120, help="Max tokens in generated joke (default 120).")
    ap.add_argument("--min_length", type=int, default=15, help="Min tokens before allowing early stop (default 15).")
    ap.add_argument("--num", type=int, default=1, help="Number of jokes to generate per prompt (default 1).")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (optional).")
    return ap.parse_args()

def main():
    args = parse_args()
    print("Loading dataset from:", args.dataset)
    texts = load_texts_from_csv(args.dataset, text_column="text")
    print("Loaded {} texts.".format(len(texts)))
    if not texts:
        print("No texts found in dataset. Exiting.", file=sys.stderr)
        return

    print("Building models up to N={}...".format(args.N))
    models = MarkovModels(N=args.N)
    models.build_from_texts(texts)
    print("Model built.")
    print("Unigram tokens:", len(models.unigram_counts))
    print("Number of {}-gram states: {}".format(args.N, len(models.models[args.N])))

    print("\nInteractive mode:")
    print("Enter comma-separated topic keywords (e.g. 'программист, начальник') and press Enter.")
    print("Leave empty to generate with a random start.")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            try:
                raw = input("prompt> ")
            except EOFError:
                break
            kws = [p.strip() for p in raw.split(",")] if raw.strip() else []
            for i in range(args.num):
                jok = generate_one(models_obj=models,
                                   prompt_words=kws,
                                   N=args.N,
                                   threshold=args.threshold,
                                   T=args.T,
                                   k_check=args.k_check,
                                   L_recent=args.L_recent,
                                   alpha=args.alpha,
                                   p_force=args.p_force,
                                   max_length=args.max_length,
                                   min_length=args.min_length,
                                   lcp_threshold=args.lcp_threshold,
                                   seed=args.seed)
                print("\n--- joke #{} ---".format(i+1))
                print(jok)
                print("---------------\n")
    except KeyboardInterrupt:
        print("\nExited by user (Ctrl+C).")
        return

if __name__ == "__main__":
    main()
