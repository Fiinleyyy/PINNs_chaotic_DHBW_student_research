# PINNs_chaotic_DHBW_student_research

Repository for a student research project at DHBW Ravensburg, Campus Friedrichshafen, exploring Physics-Informed Neural Networks (PINNs) for solving inverse problems in chaotic systems using the Lorenz system as a case study.

## Authors

This project was conducted as part of a student research collaboration between:

- **Justin Masch| 4705095**
- **Finley Hogan | 6486412**
- **Philipp Rottweiler | 7095401**

and Prof. Dr. Jürgen Schneider

## Project Structure

The repository is organized as follows:

### `notebooks/`

Contains various Jupyter notebooks used to experiment with, research, document and evaluate the performance of PINNs on different types of system dynamics (chaotic, periodic, non-chaotic).

| Notebook File                          | Description                                                                 |
|----------------------------------------|-----------------------------------------------------------------------------|
| `data_pinn_investigation.ipynb`        | Investigates forward prediction using data-driven PINNs for noisy, gappy, or partial observations. |
| `inverse_problem_investigation.ipynb`  | Evaluates inverse problem solving capabilities for non-chaotic and periodic systems. |
| `pinn_non_chaotic_investigation.ipynb` | Studies forward prediction performance of PINNs for non-chaotic dynamics.          |
| `pinn_parameter_optimisation.ipynb`    | Focuses on forward prediction with PINNs in periodic regimes.              |
| `pinn_periodic_investigation.ipynb`    | Experiments with hyperparameter optimization for chaotic system behavior.  |

### `src/`

Contains the core Python modules used in the notebooks.

| File Path                                    | Description                                                                 |
|----------------------------------------------|-----------------------------------------------------------------------------|
| `src/data_preparation/data_utils.py`         | Functions to generate reference data and simulate noisy, partial, or gappy data. |
| `src/pinn/pinn_forward_solver.py`            | PINN forward model implementation.                                          |
| `src/pinn/pinn_inverse_solver.py`            | Training and evaluation routines for inverse PINNs.                         |
| `src/pinn/neural_networks_utils.py`          | Utilities to create standard and L2-regularized neural networks.           |
| `src/pinn/pinn_utils.py`                     | Helper functions for PINN loss calculation, model building, and training.  |

## Getting Started

To run this project locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/Fiinleyyy/PINNs_chaotic_DHBW_student_research.git
cd PINNs_chaotic_DHBW_student_research
```

### 2. Set Up a Python Environment

Create a virtual environment (recommended):

```bash
python -m venv venv
```

Activate it:

- **On Windows:**
  ```bash
  .\venv\Scripts\activate
  ```
- **On macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies

Install required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Run a Notebook

Start Jupyter and open any of the project notebooks:

```bash
jupyter notebook
```

Then in the browser UI, navigate to the `notebooks/` directory and open e.g. `pinn_non_chaotic.ipynb`.

---

