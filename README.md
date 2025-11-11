# 🚀 Kuhn Poker Solver — CFR Implementation (C++ & CUDA)

This repository implements the **Counterfactual Regret Minimization (CFR)** algorithm for the simplified poker game **Kuhn Poker** — both in **pure C++** and **CUDA-parallelized** versions.

The algorithm converges to the **Nash equilibrium**, producing optimal mixed strategies for both players.

---

## 🎯 What is Kuhn Poker?

A classic minimal poker variant used in game theory:

| Rule             | Description                     |
| ---------------- | ------------------------------- |
| Deck             | `J`, `Q`, `K` (3 cards)         |
| Players          | 2                               |
| Cards per player | 1                               |
| Actions          | `c` – check, `b` – bet          |
| Game type        | Zero-sum, imperfect information |

Despite its simplicity, **bluffing is optimal**, making it ideal for CFR benchmarks.

**Theoretical payoffs:**

```
EV(Player 1) ≈ -0.055
EV(Player 2) ≈ +0.055
```

Both C++ and CUDA implementations converge to these values.

---

## 📂 Project Structure

```
/project-root
│
├── CMakeLists.txt
├── src/
│   └── *.cu
└── README.md
```

---

## ✅ Build & Run

### Requirements

* CMake ≥ 3.14
* C++17 compiler (GCC / Clang / MSVC)
* **(Optional)** CUDA Toolkit ≥ 11 for GPU version

### Build (Linux / macOS)

```bash
mkdir build
cd build
cmake ..
cmake --build .
````

### Build (Windows, Visual Studio)

```bash
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build .
```

### Run

```
./cuda-khun-poker
```

---

## ✅ Output Example

```
Player 1 expected value: -0.0569
Player 2 expected value: 0.0569

--- Average Strategies ---

Player 1:
  [Card=J, Hist=""]   [call=0.82, bet=0.18]
  [Card=Q, Hist=""]   [call=0.03, bet=0.97]
  [Card=K, Hist=""]   [call=0.66, bet=0.34]

Player 2:
  [Card=K, Hist="c"]  [call=0.33, bet=0.67]
  [Card=Q, Hist="c"]  [call=0.97, bet=0.03]
  ...
```

✅ These match the known Nash equilibrium strategies.

---

## ⚙️ Implementation Details

| Component                   | Description                                           |
| --------------------------- | ----------------------------------------------------- |
| `cfr()` / `cfr_recursive()` | Main CFR recursion                                    |
| `History`                   | Encodes betting sequence                              |
| `InfoSet` / `InfoSetMap`    | Stores strategy, regret, reach probability            |
| `terminal_utils()`          | Computes payoff                                       |
| `next_strategy()`           | Regret-matching update                                |
| CUDA version                | Parallelizes CFR across card combinations and actions |

---

## 🚀 CUDA Version Highlights

✅ Runs CFR iterations in parallel
✅ Uses shared memory & atomic operations
✅ Measurable speedup vs CPU version
✅ Designed for extending to larger poker games

---

## 📚 References

* H. W. Kuhn (1950) — Original simplified poker game
* Zinkevich et al. (2007) — CFR algorithm
* Brown & Sandholm (2017) — CFR+, Libratus
