uv run cmp_original.py --exp='syn1' --dst='syn1original' --lr=.3 --eps=.5 > /dev/null
uv run cmp_original.py --exp='syn4' --dst='syn4original' --lr=.5 --eps=.5 > /dev/null
uv run cmp_original.py --exp='syn5' --dst='syn5original' --lr=.3 --eps=.5 > /dev/null

uv run evaluate.py --exp='syn1' --dst='syn1original.pkl' > ../evaluation.txt
uv run evaluate.py --exp='syn4' --dst='syn4original.pkl' >> ../evaluation.txt
uv run evaluate.py --exp='syn5' --dst='syn5original.pkl' >> ../evaluation.txt

cat ../evaluation.txt
