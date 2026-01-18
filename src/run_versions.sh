uv run main_explain.py --exp='syn1' --dst='syn1original'   --cf_method='original'      --momentum=0.9 > /dev/null
uv run main_explain.py --exp='syn1' --dst='syn1wrapped'    --cf_method='cf_wrapped'    --momentum=0.9 > /dev/null
uv run main_explain.py --exp='syn1' --dst='syn1transposed' --cf_method='cf_transposed' --momentum=0.9 > /dev/null
# uv run main_explain.py --exp='syn1' --dst='syn1greedy'     --cf_method='greedy'        --eps=.1   > /dev/null
# uv run main_explain.py --exp='syn1' --dst='syn1bf'         --cf_method='bf'            --eps=.1   > /dev/null

# uv run main_explain.py --exp='syn2' --dst='syn2original'   --cf_method='original'      --momentum=0.9 > /dev/null
# uv run main_explain.py --exp='syn2' --dst='syn2wrapped'    --cf_method='cf_wrapped'    --momentum=0.9 > /dev/null
uv run main_explain.py --exp='syn2' --dst='syn2transposed' --cf_method='cf_transposed' --momentum=0.9 > /dev/null
# uv run main_explain.py --exp='syn2' --dst='syn2greedy'     --cf_method='greedy'        --eps=.1   > /dev/null
# uv run main_explain.py --exp='syn2' --dst='syn2bf'         --cf_method='bf'            --eps=.1   > /dev/null

# uv run main_explain.py --exp='syn4' --dst='syn4original'   --cf_method='original'          > /dev/null
# uv run main_explain.py --exp='syn4' --dst='syn4wrapped'    --cf_method='cf_wrapped'        > /dev/null
uv run main_explain.py --exp='syn4' --dst='syn4transposed' --cf_method='cf_transposed'     > /dev/null
# uv run main_explain.py --exp='syn4' --dst='syn4greedy'     --cf_method='greedy' --eps=.1   > /dev/null
# uv run main_explain.py --exp='syn4' --dst='syn4bf'         --cf_method='bf'     --eps=.1   > /dev/null

# uv run main_explain.py --exp='syn5' --dst='syn5original'   --cf_method='original'          > /dev/null
# uv run main_explain.py --exp='syn5' --dst='syn5wrapped'    --cf_method='cf_wrapped'        > /dev/null
uv run main_explain.py --exp='syn5' --dst='syn5transposed' --cf_method='cf_transposed'     > /dev/null
# uv run main_explain.py --exp='syn5' --dst='syn5greedy'     --cf_method='greedy' --eps=.1   > /dev/null
# uv run main_explain.py --exp='syn5' --dst='syn5bf'         --cf_method='bf'     --eps=.1   > /dev/null

uv run evaluate.py --exp='syn1' --dst='syn1original.pkl'   >  ../results/evaluate/syn1.txt
uv run evaluate.py --exp='syn1' --dst='syn1wrapped.pkl'    >> ../results/evaluate/syn1.txt
uv run evaluate.py --exp='syn1' --dst='syn1transposed.pkl' >> ../results/evaluate/syn1.txt
uv run evaluate.py --exp='syn1' --dst='syn1greedy.pkl'     >> ../results/evaluate/syn1.txt
uv run evaluate.py --exp='syn1' --dst='syn1bf.pkl'         >> ../results/evaluate/syn1.txt

uv run evaluate.py --exp='syn2' --dst='syn2original.pkl'   >  ../results/evaluate/syn2.txt
uv run evaluate.py --exp='syn2' --dst='syn2wrapped.pkl'    >> ../results/evaluate/syn2.txt
uv run evaluate.py --exp='syn2' --dst='syn2transposed.pkl' >> ../results/evaluate/syn2.txt
uv run evaluate.py --exp='syn2' --dst='syn2greedy.pkl'     >> ../results/evaluate/syn2.txt
uv run evaluate.py --exp='syn2' --dst='syn2bf.pkl'         >> ../results/evaluate/syn2.txt

uv run evaluate.py --exp='syn4' --dst='syn4original.pkl'   >  ../results/evaluate/syn4.txt
uv run evaluate.py --exp='syn4' --dst='syn4wrapped.pkl'    >> ../results/evaluate/syn4.txt
uv run evaluate.py --exp='syn4' --dst='syn4transposed.pkl' >> ../results/evaluate/syn4.txt
uv run evaluate.py --exp='syn4' --dst='syn4greedy.pkl'     >> ../results/evaluate/syn4.txt
uv run evaluate.py --exp='syn4' --dst='syn4bf.pkl'         >> ../results/evaluate/syn4.txt

uv run evaluate.py --exp='syn5' --dst='syn5original.pkl'   >  ../results/evaluate/syn5.txt
uv run evaluate.py --exp='syn5' --dst='syn5wrapped.pkl'    >> ../results/evaluate/syn5.txt
uv run evaluate.py --exp='syn5' --dst='syn5transposed.pkl' >> ../results/evaluate/syn5.txt
uv run evaluate.py --exp='syn5' --dst='syn5greedy.pkl'     >> ../results/evaluate/syn5.txt
uv run evaluate.py --exp='syn5' --dst='syn5bf.pkl'         >> ../results/evaluate/syn5.txt
