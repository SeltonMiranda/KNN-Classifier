#!/bin/bash

# Verifica se o número de argumentos são 3
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 FILENAME TOTAL_LINES PERCENTAGE_FOR_TRAIN"
    echo "Example: $0 myfile.txt 10000 70"
    exit 1
fi

FILENAME=$1
TOTAL_LINES=$2
PERCENTAGE_FOR_TRAIN=$3

# Verifica se o arquivo existe
if [ ! -f "$FILENAME" ]; then
    echo "Error: File '$FILENAME' does not exist."
    exit 1
fi

# Total de linhas do arquivo
LINES_IN_FILE=$(wc -l < "$FILENAME")

# Verifica se a quantidade total de linhas do argumento não ultrapassa o total de linhas do arquivo
if [ "$TOTAL_LINES" -gt "$LINES_IN_FILE" ]; then
    echo "Error: File only has $LINES_IN_FILE lines, cannot select $TOTAL_LINES lines."
    exit 1
fi

# Calcula o número de linhas para o conjunto de treino e teste
NUM_LINES_FOR_TRAIN=$(( TOTAL_LINES * PERCENTAGE_FOR_TRAIN / 100 ))
NUM_LINES_FOR_TEST=$(( TOTAL_LINES - NUM_LINES_FOR_TRAIN ))

# Aleatoriamente embaralha as linhas do arquivo e seleciona as linhas para treino e teste
echo "Selecting $NUM_LINES_FOR_TRAIN lines for training and $NUM_LINES_FOR_TEST lines for testing."
shuf -n "$NUM_LINES_FOR_TRAIN" "$FILENAME" > train_subset.txt
shuf -n "$NUM_LINES_FOR_TEST" "$FILENAME" > test_subset.txt

echo "Train subset saved to 'train_subset.txt'."
echo "Test subset saved to 'test_subset.txt'."

