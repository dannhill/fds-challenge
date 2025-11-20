-----

# 🏆 Pokémon Battles Prediction - FDS Kaggle Challenge

Repository for the **FDS Kaggle Competition 2025-2026** (Sapienza University of Rome) solution for the team DJI.

## 📌 Introduction

The goal of this challenge is to train a Machine Learning model capable of predicting the winner of a Pokémon battle using only data from the first **30 turns** of gameplay.

The problem is a **Binary Classification** task ($y \in \{0, 1\}$), where the target is to determine if "Player 1" won. The main challenge lies in the nature of the data: large `.jsonl` (JSON Lines) files containing nested structures describing:

  * **Static Features:** Team composition, base stats, moves, and items (known at turn 0).
  * **Dynamic Features:** Battle timeline, HP changes, status conditions, and switches occurring within the first 30 turns.

## 🚀 Architecture & C Optimization

[WIP]

### Compatibility

[WIP]

## 🛠️ Installation & Compilation

### 1\. Clone the repository

```bash
git clone https://github.com/dannhill/fds-challenge.git
cd fds-challenge
```

### 2\. Compile the C source code

**Windows (CMD/PowerShell):**

```cmd
cl main.c libs\cJSON\cJSON.c /Fe:preprocessor.exe /I libs\cJSON /I libs\uthash /O2 /arch:AVX2 /fp:fast /W3 /EHsc
```

### 3\. Python Environment

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

## ▶️ Usage

### Step 1: Data Preprocessing

Run the C executable to convert the raw data into training-ready datasets.
*Ensure you have downloaded the competition files into the `data/` folder.*

```bash
# Usage: ./preprocess.exe
```
[WIP]
## 👥 Authors

  * **[Danilo Medas]**
  * **[Jassahib Singh]**
  * **[Muhammad Imad Aziz Khan]**

-----
