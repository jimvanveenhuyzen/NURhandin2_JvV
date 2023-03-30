#!/bin/bash

python3 NURhandin2_1.py > NURhandin2_problem1a.txt
python3 NURhandin2_1.py > NURhandin2_problem1d.txt
python3 NURhandin2_2

echo "Generating the pdf"

pdflatex handin2_JvV_s2272881.tex


