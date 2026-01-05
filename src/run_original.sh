uv run cmp_original.py --exp='syn1' --dst='syn1original' --lr=.1 --eps=1 --momentum=.9 > /dev/null
uv run cmp_original.py --exp='syn4' --dst='syn4original' --lr=.1 --eps=1 > /dev/null
uv run cmp_original.py --exp='syn5' --dst='syn5original' --lr=.1 --eps=1 > /dev/null

uv run evaluate.py --exp='syn1' --dst='syn1original.pkl' > ../evaluation.txt
uv run evaluate.py --exp='syn4' --dst='syn4original.pkl' >> ../evaluation.txt
uv run evaluate.py --exp='syn5' --dst='syn5original.pkl' >> ../evaluation.txt

cat ../evaluation.txt
