# Explain and evaluate using old framework gradients.

uv run ../src/main_explain.py --exp='syn1' --dst='syn1original' --cf_method='original'   --momentum=0.9 > /dev/null
uv run ../src/main_explain.py --exp='syn1' --dst='syn1sparsecf' --sparse=True --momentum=0.9 > /dev/null

uv run ../src/main_explain.py --exp='syn2' --dst='syn2original' --cf_method='original'   --momentum=0.9 > /dev/null
uv run ../src/main_explain.py --exp='syn2' --dst='syn2sparsecf' --sparse=True --momentum=0.9 > /dev/null

uv run ../src/main_explain.py --exp='syn4' --dst='syn4original'   --cf_method='original'          > /dev/null
uv run ../src/main_explain.py --exp='syn4' --dst='syn4sparsecf' --sparse=True > /dev/null

uv run ../src/main_explain.py --exp='syn5' --dst='syn5original'   --cf_method='original'          > /dev/null
uv run ../src/main_explain.py --exp='syn5' --dst='syn5sparsecf' --sparse=True > /dev/null

uv run ../src/evaluate.py --exp='syn1' --dst='syn1original.pkl' >  ../results/evaluate/syn1.txt
uv run ../src/evaluate.py --exp='syn1' --dst='syn1sparsecf.pkl' >> ../results/evaluate/syn1.txt

uv run ../src/evaluate.py --exp='syn2' --dst='syn2original.pkl' >  ../results/evaluate/syn2.txt
uv run ../src/evaluate.py --exp='syn2' --dst='syn2sparsecf.pkl' >> ../results/evaluate/syn2.txt

uv run ../src/evaluate.py --exp='syn4' --dst='syn4original.pkl' >  ../results/evaluate/syn4.txt
uv run ../src/evaluate.py --exp='syn4' --dst='syn4sparsecf.pkl' >> ../results/evaluate/syn4.txt

uv run ../src/evaluate.py --exp='syn5' --dst='syn5original.pkl' >  ../results/evaluate/syn5.txt
uv run ../src/evaluate.py --exp='syn5' --dst='syn5sparsecf.pkl' >> ../results/evaluate/syn5.txt
