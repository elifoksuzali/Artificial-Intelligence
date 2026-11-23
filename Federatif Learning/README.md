# Federatif Learning (Federated Learning)

Federated / distributed learning experiments for **predictive maintenance** on the NASA **C-MAPSS turbofan engine degradation** datasets (**FD001–FD004**). Models are trained across multiple simulated workers without centralizing the raw data, covering both synchronous and asynchronous aggregation.

## Contents

| File | Description |
|------|-------------|
| `FD001_Distributed_LSTM_10_Workers_alpha_004_epoch_01.ipynb` | Distributed LSTM across 10 workers on FD001 (RUL regression) |
| `FEDSVM_10_workers_FD001.ipynb` | Federated SVM, 10 workers, synchronous (FD001) |
| `FedSVM_10_workers_asynchronous_FD001.ipynb` | Asynchronous FedSVM — FD001 |
| `FedSVM_10_workers_asynchronous_FD002.ipynb` | Asynchronous FedSVM — FD002 |
| `FedSVM_10_workers_asynchronous_FD003.ipynb` | Asynchronous FedSVM — FD003 |
| `FedSVM_10_workers_asynchronous_FD004.ipynb` | Asynchronous FedSVM — FD004 |
| `federe.ipynb` / `federe.py` | Core federated-learning implementation |
| `final.py` | Final consolidated script |

## Concepts

- **Dataset:** NASA C-MAPSS (FD001–FD004) run-to-failure turbofan sensor data; typical task is Remaining Useful Life (RUL) estimation.
- **Workers:** 10 simulated clients holding partitioned data.
- **Aggregation:** synchronous and **asynchronous** federated averaging variants.
- **Models:** LSTM (sequence model) and SVM (federated).

## Stack

Python · PyTorch / scikit-learn · NumPy / pandas · Google Colab

## Notes

Download the C-MAPSS dataset separately and point the notebooks to the FD00x train/test files before running.
