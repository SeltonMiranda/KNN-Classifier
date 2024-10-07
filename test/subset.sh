#!/bin/bash

# Verifica se o número de argumentos são 2
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 filename number_of_lines_to_select"
    echo "Example: $0 myfile.txt 10"
    exit 1
fi

filename=$1
num_lines_to_select=$2

# Verifica se o arquivo existe
if [ ! -f "$filename" ]; then
    echo "Error: File '$filename' does not exist."
    exit 1
fi

# Total de linhas do arquivo
total_lines=$(wc -l < "$filename")

# Verifica se a quantidade de linhas do argumento não ultrapassa o total de linhas do arquivo
if [ "$num_lines_to_select" -gt "$total_lines" ]; then
    echo "Error: File only has $total_lines lines, cannot select $num_lines_to_select lines."
    exit 1
fi

# Aleatoriamente embaralha as linhas do arquivo
shuf -n "$num_lines_to_select" "$filename"
