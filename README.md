# Learning Differential Physics — PINNs vs FDM for PDEs

A scientific computing + machine learning project that solves the **1D Heat Equation** using:

* Classical numerical methods (**Finite Difference Method - FDM**)
* A modern deep learning approach (**Physics-Informed Neural Networks - PINNs**)

The project demonstrates how physics can be enforced directly into neural networks, enabling them to solve differential equations **without labeled data**.

---

## Why This Project

This project aims to bridge **mathematics, physics, and machine learning** by:

* Implementing a stable numerical PDE solver (FDM)
* Building a PINN that learns solutions using physics constraints
* Comparing both methods against an **analytical ground truth**
* Evaluating accuracy using quantitative metrics (MSE, L2 error)
* Visualizing spatio-temporal behavior of solutions

This is a **research-style workflow**, not just a model implementation.

---

## Problem Definition

We solve the **1D Heat Equation**:

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
$$

Where:

* $u(x,t)$ = temperature distribution
* $\alpha$ = diffusion coefficient

### Domain and Conditions

* Spatial domain: $x \in [0,1]$
* Time domain: $t \in [0,1]$
* Initial condition: $u(x,0) = \sin(\pi x)$
* Boundary conditions: $u(0,t) = 0, \quad u(1,t) = 0$

---

## Analytical Solution (Ground Truth)

For this setup, the exact solution is:

$$
u(x,t) = e^{-\pi^2 \alpha t} \sin(\pi x)
$$

This allows **direct validation** of both FDM and PINN results.

---

## Methods

### 1) Finite Difference Method (FDM)

A classical numerical approach using the **Forward-Time Central-Space (FTCS)** scheme.

#### Discretization

* Time derivative:

  $$
  u_t \approx \frac{u_i^{n+1} - u_i^n}{\Delta t}
  $$

* Second spatial derivative:

  $$
  u_{xx} \approx \frac{u_{i+1}^n - 2u_i^n + u_{i-1}^n}{\Delta x^2}
  $$

#### Update Rule

$$
u_i^{n+1} = u_i^n + r \left(u_{i+1}^n - 2u_i^n + u_{i-1}^n \right)
$$

where:

$$
r = \frac{\alpha \Delta t}{\Delta x^2}
$$

#### Stability Condition

$$
\Delta t \leq \frac{\Delta x^2}{2\alpha}
$$

FDM serves as a **trusted numerical baseline**.

---

### 2) Physics-Informed Neural Network (PINN)

Instead of solving the PDE directly, we train a neural network:

$$
u_\theta(x,t)
$$

to satisfy physics constraints.

#### Model Architecture

* Fully connected neural network (MLP)
* Input: $(x, t)$
* Output: $u(x,t)$
* Activation: Tanh
* Depth: 3 layers
* Width: 20–30 neurons

---

## Loss Function (Core Idea)

The PINN is trained using **physics + constraints**, not labeled data.

### 1) PDE Residual Loss

$$
\mathcal{L}_{PDE} = \mathbb{E}[(u_t - \alpha u_{xx})^2]
$$

### 2) Initial Condition Loss

$$
\mathcal{L}_{IC} = \mathbb{E}[(u(x,0) - \sin(\pi x))^2]
$$

### 3) Boundary Condition Loss

$$
\mathcal{L}_{BC} = \mathbb{E}[(u(0,t))^2 + (u(1,t))^2]
$$

### Total Loss

$$
\mathcal{L} = \mathcal{L}_{PDE} + \mathcal{L}_{IC} + \mathcal{L}_{BC}
$$

---

## Key Insight (Why PINNs Work)

Instead of learning from data:

* The model learns by **minimizing physics violations**
* Uses **automatic differentiation (PyTorch autograd)** to compute:

  * $u_t$
  * $u_{xx}$

This makes the network **physics-aware by design**.

---

## Training Strategy

* Optimizer: Adam
* Learning rate: $5 \times 10^{-4}$
* Epochs: 3000
* Sampling:

  * Interior points → PDE loss
  * Initial points → IC loss
  * Boundary points → BC loss

---

## Results

### Final Metrics

* **MSE (PINN vs Exact):** ~4.6e-6
* **Relative L2 Error:** ~0.0046 (~0.46%)

### Interpretation

* PINN achieves **>99.5% accuracy**
* Matches both:

  * Numerical solution (FDM)
  * Analytical solution

---

## Visual Results

### Heatmaps

* FDM and PINN produce nearly identical spatio-temporal patterns
* Shows correct diffusion behavior

### Training Loss

* Loss decreases by several orders of magnitude
* IC converges fastest
* PDE residual converges slower (expected)

### Time Slice Comparison

* PINN predictions overlap closely with reference curves
* Confirms correct temporal evolution

---

## Project Structure

```text
learning-differential-physics/
│
├── src/
│   ├── pinn_model.py
│   ├── fdm_solver.py
│   ├── train.py
│   ├── utils.py
│   ├── visualization.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── outputs/
│   ├── metrics.csv
│   ├── loss_history.csv
│   ├── *.png plots
│   └── pinn_model.pt
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd learning-differential-physics
```

### 2. Setup Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Experiment

Use the notebook:

```bash
jupyter notebook notebooks/experiments.ipynb
```

---

## CPU vs GPU

* Designed to run on CPU (tested locally)
* PINNs can be slow on CPU due to autograd
* Optional: Use **Google Colab GPU** for faster training

---

## Key Takeaways

* FDM solves PDEs numerically via discretization
* PINNs solve PDEs by **learning physics constraints**
* PINNs are:

  * Mesh-free
  * Data-efficient
  * Computationally heavier

---

## Limitations

* PINN training is slower than FDM
* Requires careful loss balancing
* Only the 1D heat equation is implemented

---

## Future Improvements

* Extend to 2D/3D PDEs
* Adaptive sampling strategies
* Loss weighting experiments
* Hybrid FDM + PINN approaches
* GPU-accelerated training pipelines

---

## License

This project is provided as-is for educational purposes. Feel free to use, modify, and distribute.

## Author

**Sanket Motagi** | Created: March 2026  
GitHub: [@Sanku-ctrl](https://github.com/Sanku-ctrl)

## Contact & Contributing

Found a bug or have suggestions? Feel free to:
- Open an issue on GitHub
- Submit a pull request
- Contact via email

---