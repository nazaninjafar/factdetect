# FactDetect


FactDetect is an effective approach for decomposing evidence sentences into shorter, more relevant sentences. Our method prioritizes relevance to the claim and importance for the verdict, based on the connection between evidence and the claim.


## Usage

To run the model in zero-shot setting, use the following command:

```bash
python factdetect/prompting.py --llm_model_name [LLM Checkpoint] \
            --test_file [TEST FILE] \
            --corpus [CORPUS] \
            --prompt_type factdetect \
            --outfilename [OUTPUT FILE]
   
