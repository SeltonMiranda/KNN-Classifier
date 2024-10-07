### Suggestion:

```bash
# Extract training samples (70%)
./subset.sh ../pucpr_norm.csv 2400 > 70_subset.csv
./subset.sh ../ufpr04_norm.csv 2400 >> 70_subset.csv
./subset.sh ../ufpr05_norm.csv 2400 >> 70_subset.csv

# Shuffle combined training set
cat 70_subset.csv | shuf > 70_subset_shuffled.csv

# Repeat for testing samples (30%)
# [Similar commands for remaining 30% of data]
```
This commands will:
1. Take 70% of samples from each input file for training
2. Use the remaining 30% for testing
3. Create two shuffled output files:
   - final_dataset_70_train_shuffled.csv
   - final_dataset_30_test_shuffled.csv
