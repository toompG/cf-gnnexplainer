uv run cmp_original.py --exp='syn1' --dst='syn1greedy' --cf='greedy' > /dev/null
uv run cmp_original.py --exp='syn1' --dst='syn1bf'     --cf='bf' > /dev/null
uv run cmp_original.py --exp='syn4' --dst='syn4greedy' --cf='greedy' > /dev/null
uv run cmp_original.py --exp='syn4' --dst='syn4bf'     --cf='bf' > /dev/null
uv run cmp_original.py --exp='syn5' --dst='syn5greedy' --cf='greedy' > /dev/null
uv run cmp_original.py --exp='syn5' --dst='syn5bf'     --cf='bf' > /dev/null

uv run evaluate.py --exp='syn1' --dst='syn1greedy.pkl' > ../run_versions.txt
uv run evaluate.py --exp='syn1' --dst='syn1bf.pkl' >> ../run_versions.txt
uv run evaluate.py --exp='syn4' --dst='syn4greedy.pkl' >> ../run_versions.txt
uv run evaluate.py --exp='syn4' --dst='syn4bf.pkl' >> ../run_versions.txt
uv run evaluate.py --exp='syn5' --dst='syn5greedy.pkl' >> ../run_versions.txt
uv run evaluate.py --exp='syn5' --dst='syn5bf.pkl' >> ../run_versions.txt

cat ../run_versions.txt
