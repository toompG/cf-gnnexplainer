# Explain and evaluate datasets using new framework's gradients

uv run ../src/main_explain.py --exp='syn1' --dst='syn1sparsecf'     --sparse=True --momentum=0.9 > /dev/null
uv run ../src/main_explain.py --exp='syn1' --dst='syn1sparsegreedy' --sparse=True --cf_method='greedy' > /dev/null
uv run ../src/main_explain.py --exp='syn1' --dst='syn1sparsebf'     --sparse=True --cf_method='bf'      > /dev/null

uv run ../src/main_explain.py --exp='syn2' --dst='syn2sparsecf'     --sparse=True > /dev/null
uv run ../src/main_explain.py --exp='syn2' --dst='syn2sparsegreedy' --sparse=True --cf_method='greedy' > /dev/null
uv run ../src/main_explain.py --exp='syn2' --dst='syn2sparsebf'     --sparse=True --cf_method='bf'     > /dev/null

uv run ../src/main_explain.py --exp='syn4' --dst='syn4sparsecf'     --sparse=True > /dev/null
uv run ../src/main_explain.py --exp='syn4' --dst='syn4sparsegreedy' --sparse=True --cf_method='greedy' > /dev/null
uv run ../src/main_explain.py --exp='syn4' --dst='syn4sparsebf'     --sparse=True --cf_method='bf'     > /dev/null

uv run ../src/main_explain.py --exp='syn5' --dst='syn5sparsecf'     --sparse=True > /dev/null
uv run ../src/main_explain.py --exp='syn5' --dst='syn5sparsegreedy' --sparse=True --cf_method='greedy' > /dev/null
uv run ../src/main_explain.py --exp='syn5' --dst='syn5sparsebf'     --sparse=True --cf_method='bf'     > /dev/null

uv run ../src/evaluate.py --exp='syn1' --dst='syn1sparsecf' > ../results/evaluate/syn1_sparse.txt
uv run ../src/evaluate.py --exp='syn1' --dst='syn1sparsegreedy' >> ../results/evaluate/syn1_sparse.txt
uv run ../src/evaluate.py --exp='syn1' --dst='syn1sparsebf' >> ../results/evaluate/syn1_sparse.txt

uv run ../src/evaluate.py --exp='syn2' --dst='syn2sparsecf' > ../results/evaluate/syn2_sparse.txt
uv run ../src/evaluate.py --exp='syn2' --dst='syn2sparsegreedy' >> ../results/evaluate/syn2_sparse.txt
uv run ../src/evaluate.py --exp='syn2' --dst='syn2sparsebf' >> ../results/evaluate/syn2_sparse.txt

uv run ../src/evaluate.py --exp='syn4' --dst='syn4sparsecf' > ../results/evaluate/syn4_sparse.txt
uv run ../src/evaluate.py --exp='syn4' --dst='syn4sparsegreedy' >> ../results/evaluate/syn4_sparse.txt
uv run ../src/evaluate.py --exp='syn4' --dst='syn4sparsebf' >> ../results/evaluate/syn4_sparse.txt

uv run ../src/evaluate.py --exp='syn5' --dst='syn5sparsecf' > ../results/evaluate/syn5_sparse.txt
uv run ../src/evaluate.py --exp='syn5' --dst='syn5sparsegreedy' >> ../results/evaluate/syn5_sparse.txt
uv run ../src/evaluate.py --exp='syn5' --dst='syn5sparsebf' >> ../results/evaluate/syn5_sparse.txt
