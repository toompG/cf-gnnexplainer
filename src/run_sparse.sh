uv run main_explain.py --exp='syn1'   --dst='syn1sparsecf'     --sparse=True --momentum=0.9 > /dev/null
uv run main_explain.py --exp='syn1'   --dst='syn1sparsegreedy' --sparse=True --cf_method='greedy' --eps=.1   > /dev/null
uv run main_explain.py --exp='syn1'   --dst='syn1sparsebf'     --sparse=True --cf_method='bf'     --eps=.1   > /dev/null

uv run main_explain.py --exp='syn2'   --dst='syn2sparsecf'     --sparse=True > /dev/null
uv run main_explain.py --exp='syn2'   --dst='syn2sparsegreedy' --sparse=True --cf_method='greedy' --eps=.1   > /dev/null
uv run main_explain.py --exp='syn2'   --dst='syn2sparsebf'     --sparse=True --cf_method='bf'     --eps=.1   > /dev/null

uv run main_explain.py --exp='syn4'   --dst='syn4sparsecf'     --sparse=True > /dev/null
uv run main_explain.py --exp='syn4'   --dst='syn4sparsegreedy' --sparse=True --cf_method='greedy' --eps=.1   > /dev/null
uv run main_explain.py --exp='syn4'   --dst='syn4sparsebf'     --sparse=True --cf_method='bf'     --eps=.1   > /dev/null

uv run main_explain.py --exp='syn5'   --dst='syn5sparsecf'     --sparse=True > /dev/null
uv run main_explain.py --exp='syn5'   --dst='syn5sparsegreedy' --sparse=True --cf_method='greedy' --eps=.1   > /dev/null
uv run main_explain.py --exp='syn5'   --dst='syn5sparsebf'     --sparse=True --cf_method='bf'     --eps=.1   > /dev/null

uv run evaluate.py --exp='syn1' --dst='syn1sparsecf' --sparse=True > ../results/evaluate/syn1_sparse.txt
uv run evaluate.py --exp='syn1' --dst='syn1sparsegreedy' --sparse=True >> ../results/evaluate/syn1_sparse.txt
uv run evaluate.py --exp='syn1' --dst='syn1sparsebf' --sparse=True >> ../results/evaluate/syn1_sparse.txt

uv run evaluate.py --exp='syn2' --dst='syn2sparsecf' --sparse=True > ../results/evaluate/syn2_sparse.txt
uv run evaluate.py --exp='syn2' --dst='syn2sparsegreedy' --sparse=True >> ../results/evaluate/syn2_sparse.txt
uv run evaluate.py --exp='syn2' --dst='syn2sparsebf' --sparse=True >> ../results/evaluate/syn2_sparse.txt

uv run evaluate.py --exp='syn4' --dst='syn4sparsecf' --sparse=True > ../results/evaluate/syn4_sparse.txt
uv run evaluate.py --exp='syn4' --dst='syn4sparsegreedy' --sparse=True >> ../results/evaluate/syn4_sparse.txt
uv run evaluate.py --exp='syn4' --dst='syn4sparsebf' --sparse=True >> ../results/evaluate/syn4_sparse.txt

uv run evaluate.py --exp='syn5' --dst='syn5sparsecf' --sparse=True > ../results/evaluate/syn5_sparse.txt
uv run evaluate.py --exp='syn5' --dst='syn5sparsegreedy' --sparse=True >> ../results/evaluate/syn5_sparse.txt
uv run evaluate.py --exp='syn5' --dst='syn5sparsebf' --sparse=True >> ../results/evaluate/syn5_sparse.txt
