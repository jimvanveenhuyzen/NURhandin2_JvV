#!/bin/bash

python3 NURhandin2_1.py > NURhandin2problem1.txt
python3 NURhandin2_2.py > NURhandin2problem2.txt

echo "Generating the pdf"

pdflatex handin2_JvV_s2272881.tex


