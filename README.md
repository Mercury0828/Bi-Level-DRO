# Bi-Level Distributionally Robust Optimization (DRO) System

## 📋 Overview

This project implements a Bi-Level Distributionally Robust Optimization (DRO) system for supply chain network optimization. The system adopts a hybrid Python-C architecture, compiling core optimization algorithms into high-performance C extension modules, providing significant performance improvements and algorithm protection.

## 🚀 Quick Start

### Requirements

- Python 3.11+
- NumPy 1.26+
- Matplotlib
- Pandas
- C Compiler (Windows: MSVC, Linux/Mac: GCC)

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/Mercury0828/Bi-Level-DRO
cd bi_level_dro/src/core
```

2. **Install Python Dependencies**
```bash
pip install numpy matplotlib pandas
```

3. **Run the Main Program**
```bash
python main.py
```

## 📁 Project Structure

```
core/
├── core_algorithm.c        # C extension source code (core optimization algorithms)
├── core_algorithm.pyd      # Compiled C extension (Windows)
├── core_algorithm_wrap.py  # Python wrapper (provides class interfaces)
├── dataset_generation.py   # Data generation module
├── main.py                 # Main execution script
├── utils.py               # Utility functions
└── visualization.py       # Visualization module
visualization/              # Figure saving direction
```


## 📄 License

MIT License

---

*Last Updated: August 17, 2025*
