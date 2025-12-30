uv run cmp_original.py --exp='syn1' --dst='syn1cforig' --cf='original' --lr=.1 --momentum=.9 > /dev/null
uv run cmp_original.py --exp='syn4' --dst='syn4cforig' --cf='original' --lr=.1 > /dev/null
uv run cmp_original.py --exp='syn5' --dst='syn5cforig' --cf='original' --lr=.1 > /dev/null

uv run evaluate.py --exp='syn1' --dst='syn1cforig.pkl' > ../truorig.txt
uv run evaluate.py --exp='syn4' --dst='syn4cforig.pkl' >> ../truorig.txt
uv run evaluate.py --exp='syn5' --dst='syn5cforig.pkl' >> ../truorig.txt

cat ../truorig.txt