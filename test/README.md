### Suggestion:

Using 10.000 samples to test the algorithm

`
$ ./subset.sh ../pucpr_norm.csv 2400 > 70_subset.csv
$ ./subset.sh ../ufpr04_norm.csv 2400 >> 70_subset.csv
$ ./subset.sh ../ufpr05_norm.csv 2400 >> 70_subset.csv
$ cat 70_subset.csv | shuf > 70_subset_shuffled.csv
`
Do the same for the lefting 30% samples.
Or create a general script which does that automatically by yourself
