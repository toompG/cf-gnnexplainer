uv run ../src/experiments/compare_model_formats.py --exp=syn1 --method=wrapped >  syn1err.txt
uv run ../src/experiments/compare_model_formats.py --exp=syn1 --method=gcn     >> syn1err.txt
uv run ../src/experiments/compare_model_formats.py --exp=syn1 --method=gcnmm   >> syn1err.txt

uv run ../src/experiments/compare_model_formats.py --exp=syn4 --method=wrapped >  syn4err.txt
uv run ../src/experiments/compare_model_formats.py --exp=syn4 --method=gcn     >> syn4err.txt
uv run ../src/experiments/compare_model_formats.py --exp=syn4 --method=gcnmm   >> syn4err.txt

uv run ../src/experiments/compare_model_formats.py --exp=syn5 --method=wrapped >  syn5err.txt
uv run ../src/experiments/compare_model_formats.py --exp=syn5 --method=gcn     >> syn5err.txt
uv run ../src/experiments/compare_model_formats.py --exp=syn5 --method=gcnmm   >> syn5err.txt
