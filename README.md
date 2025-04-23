# clip-ood-part2
We now actually remove all sketches

## Regarding config

- `label_mapping.yaml` remains fixed and each job uses it.
- `train_data.yaml` or `eval_data.yaml` also usually remains fixed but potentially could change. 
- `eval_sc.yaml` or `train_sc.yaml` both contain arguments for job arrays. 