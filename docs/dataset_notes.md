# Dataset notes

## Default simulator

The project creates a fully synthetic dataset with these columns:

- `timestamp`: generated hourly timestamps
- `temperature`: degrees Celsius
- `vibration`: illustrative vibration level in mm/s
- `current`: illustrative current in amperes
- `rotational_speed`: RPM
- `torque`: Nm
- `tool_wear`: illustrative wear counter in minutes
- `failure`: synthetic binary label, 0 = normal, 1 = simulated failure

The target is generated from thresholds in `src/simulator.py`. This makes the project reproducible and easy to run, but it also means evaluation measures agreement with the simulator's rules, not real-world predictive validity.

## Public dataset extension

The UCI AI4I 2020 Predictive Maintenance Dataset is a possible next experiment:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Read the dataset documentation, preserve original target definitions, and document any mapping of its columns to model features.
