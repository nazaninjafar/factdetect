import os
import torch
from datasets import load_dataset

from torch.nn import functional as F

import pytorch_lightning as pl
from metrics import ClassificationMetric
from vllm import LLM, SamplingParams
import random

import torch.nn as nn

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
) 
from transformers import pipeline
from prompt_dump import few_shot_prompt_scifact_10,few_shot_prompt_scifact_10_vanilla, few_shot_prompt_scifact_10_cot

import os

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)
class Prompting:
    def __init__(self,args,llm_model_name,nei = False):
        self.args = args
        self.prompt_type = self.args.prompt_type
        self.nei = nei
        self.factuality = self.args.factuality
        
        self.llm_model_name = llm_model_name
        self.overal_inst = "This is a claim verification task."
        if self.factuality:
            self.overal_inst = ""
        if self.prompt_type =="vanilla":
            self.prompt_inst = VanillaPrompting(self.overal_inst,nei = self.nei,factuality = self.factuality).prompt_inst
        elif self.prompt_type == "cot":
            self.prompt_inst = CoTPrompting(self.overal_inst,nei = self.nei,factuality = self.factuality).prompt_inst
        elif self.prompt_type == "carp":
            self.prompt_inst = CARPPrompting(self.overal_inst,nei = self.nei).prompt_inst
        elif self.prompt_type == "factdetect":
            self.prompt_inst = FactDetectPrompting(self.overal_inst,nei=self.nei,factuality = self.factuality).prompt_inst

        if self.llm_model_name =='flant5':
                self.t5inference = T5Inference(self.args)
        if self.llm_model_name =='gpt35' or self.llm_model_name =='gpt4':
                self.gptinference = GPTInference(self.args,self.llm_model_name)
        if self.llm_model_name =='llama':
                self.llama = LlamaInference(self.args)

        if self.llm_model_name =='vicuna':
                self.llama = VicunaInference(self.args)


        if self.llm_model_name =='mistral':
                self.llama = MistralInference(self.args)


        

        
    def get_prompts(self,inputs,factdetect = False,nei = False):
        """ This is for llama models only"""
        
        for input in inputs:
            
            claim = input['claim']
            evidence = input['rationale']
            input['prompt'] = self.prompt_inst + "Input_Claim: " + "<" + claim + ">,"+ " Input_Evidence: " +"<" + evidence + ">"
            if factdetect:
                #get important facts
                
                fact_sentences = input['af_sentences']
                

                
                # af_sets = input['af_sets']
                # af_sets = [item for item in af]
                # important_sentences  = [fact_sentences[i] for i in af_sets]

                # fact_sentences = ", ".join(important_sentences)
                fact_sentences = ", ".join(fact_sentences)
                #add [] to the fact sentences
                fact_sentences = "[" + fact_sentences + "]"
                input['prompt'] = input['prompt'] + " Input_Facts: " + "<" + fact_sentences + ">"
                # input['prompt'] = input['prompt'] + " Important_Facts: " + "<" + fact_sentences + ">"

        prompt_list = [input['prompt'] for input in inputs]
        return inputs,prompt_list


    def get_all_inferences(self,inputs,nei = False):
        """ This is for llama models only"""

        if self.prompt_type == "factdetect":
            factdetect=True
        else:
            factdetect=False
        
        inputs,prompt_list = self.get_prompts(inputs,factdetect = factdetect,nei = nei)


        outputs = self.llama.get_model_predict(prompt_list)
        for output,input in zip(outputs,inputs):
            verdict = self.parse_verdict(output,nei = nei)
            if verdict is None:
                verdict = "NONE"
            input['verdict'] = verdict
            input['original_output'] = output
        return inputs
        



    
    def get_inference(self,claim,evidence,af_sentences=None,af_sets=None):
        
        self.input = "\n INPUT_Claim: " + claim + " INPUT_Evidence: " + evidence
        if self.factuality:
            self.input = "\n INPUT_Document: " + evidence + " INPUT_Statement: " + claim
            
        if self.prompt_type == "factdetect":
            if af_sentences is not None:
                #flatten af_sets 
                # af_sets = [item for sublist in af_sets for item in sublist]
                # important_sentences  = [af_sentences[i] for i in af_sets]
                af_sentences = ", ".join(af_sentences)
                # af_sentences = ", ".join(af_sentences)
                # af_sentences = "[" + af_sentences + "]"
                self.input = self.input + " Input_Facts: " + af_sentences
                # self.input = self.input + " Important_Evidence_Facts: " + af_sentences
            else:
                #raise an exception
                print("No af_sentences found in the input")
        self.input_text = self.prompt_inst + self.input

        verdict = None
        #if verdict is None, keep doing inference until you get a verdict
        
        if self.llm_model_name =='flant5':
            output = self.t5inference.get_model_predict(self.input_text)


        elif self.llm_model_name =='gpt35' or self.llm_model_name =='gpt4':
            
            output = self.gptinference.get_model_predict(self.input_text)
        
        elif self.llm_model_name =='llama':
            
            output = self.llama.get_model_predict(self.prompt_inst,claim,evidence)

            
            #parse the output to get the verdict
            
            verdict = self.parse_verdict(output)
        if verdict is None:
            verdict = "None"

        return {
            "verdict": verdict,
            "original_output": output
        }


    def clean_text(self,text):
        text = text.replace('\n','')
        text = text.replace('\t','')
        text = text.replace('\r','')
        return text


    def parse_verdict(self,output,nei = False):
        #extract after VERDICT
        verdict = None
        if nei:
        
            choices = ['CONTRADICT','NOT_ENOUGH_INFO','SUPPORT']
        else:
            choices = ['CONTRADICT','SUPPORT']

        for choice in choices:
            if choice in output:
                verdict = choice
                break

        if verdict is None:
            #raise an exception
            print("No verdict found in the output")
            

          
        return verdict
    

class T5Inference:
    def __init__(self,args,t5_encoder = "google/flan-t5-xxl"):
        super().__init__()
        self.args = args
        self.t5_encoder = t5_encoder
        self.generator = pipeline("text2text-generation",model = self.t5_encoder,device_map="auto")
        


    def get_model_predict(self,prompt):
        out = self.generator(prompt, max_length=1000, do_sample=True, temperature=0.9,  num_return_sequences=1)
    
        return out[0]['generated_text']
    
class GPTInference:
    def __init__(self,args,llm_model_name):
        super().__init__()
        self.args = args
        self.llm_model_name = llm_model_name

        if self.llm_model_name =='gpt35':
            self.model_engine = "gpt-3.5-turbo-1106"
        elif self.llm_model_name =='gpt4':
            self.model_engine = "gpt-4"

    def get_model_predict(self,prompt):
        
        completion = completion_with_backoff(model = self.model_engine,
                    messages = [{"role": "user", "content": prompt}],
                    n=1,stop=None,temperature=0)
        
        return completion.choices[0].message.content

class LlamaInference:
    def __init__(self,args):
        super().__init__()
        self.args = args
        
        self.model_id = "meta-llama/Llama-2-13b-chat-hf"
        # self.model_id = "lmsys/vicuna-13b-v1.3"
        # self.model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        self.sampling_params = SamplingParams(temperature=0.0,max_tokens= 500)
        self.llm = LLM(model=self.model_id,download_dir = CACHE_DIR)
        

    def get_model_predict(self,prompts):
        outputs = self.llm.generate(prompts, self.sampling_params)

        return [output_text.outputs[0].text for output_text in outputs]
class VicunaInference:
    def __init__(self,args):
        super().__init__()
        self.args = args
        
        # self.model_id = "meta-llama/Llama-2-13b-chat-hf"
        self.model_id = "lmsys/vicuna-13b-v1.5"
        # self.model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        self.sampling_params = SamplingParams(temperature=0.0,max_tokens= 500)
        self.llm = LLM(model=self.model_id,download_dir = CACHE_DIR)
        

    def get_model_predict(self,prompts):
        outputs = self.llm.generate(prompts, self.sampling_params)

        return [output_text.outputs[0].text for output_text in outputs]   

class MistralInference:
    def __init__(self,args):
        super().__init__()
        self.args = args
        
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.2"
      
        self.sampling_params = SamplingParams(temperature=0.0,max_tokens= 500)
        self.llm = LLM(model=self.model_id,download_dir = CACHE_DIR,dtype = torch.float16)
        

    def get_model_predict(self,prompts):
        outputs = self.llm.generate(prompts, self.sampling_params)

        return [output_text.outputs[0].text for output_text in outputs]

class VanillaPrompting():
    def __init__(self,overal_inst,few_shot=True,nei = False,factuality = False):
            super().__init__()
#inherit the prompt inst from the parent class
            self.overal_inst = overal_inst
            if nei:
                self.prompt_inst = self.overal_inst + 'Predict the verdict based on the given input Claim and the Evidence. ONLY output one of the fllowing options based on the given INPUT_Claim and the Input_Evidence: CONTRADICT, NOT_ENOUGH_INFO, SUPPORT. The output format : VERDICT: [verdict]'
            else:
                self.prompt_inst = self.overal_inst + 'Predict the verdict based on the given input Claim and the Evidence. ONLY output one of the fllowing options based on the given INPUT_Claim and the Input_Evidence: CONTRADICT, SUPPORT. The output format : VERDICT: [verdict]'
            few_shot_prompt =["""Input_Claim: <12 percent of the population in the US is suffering from diabetes.> Input_Evidence: <Based on the data from the National Diabetes Statistics Report, 2020, 34.2 million Americans, or 10.5% of the population, had diabetes. Approximately 1.6 million Americans have type 1 diabetes.>
                VERDICT: [CONTRADICT]""",
                """Input_Claim: <Rick and Morty is animated series for children and created by Dan Roiland.> Input_Evidence: <Rick and Morty is an American adult animated science fiction sitcom created by Justin Roiland and Dan Harmon for Cartoon Network's nighttime programming block Adult Swim.>
                VERDICT: [CONTRADICT]""",
                """Input_Claim: <Covid-19 is a virus that is transmitted through the air.> Input_Evidence: <COVID-19 is a respiratory disease caused by the SARS-CoV-2 virus. It is primarily spread through respiratory droplets when an infected person coughs, sneezes, or talks.>
                VERDICT: [SUPPORT]""",
                """Input_Claim: <Hunger Games is a movie inspired by the Greek myth. > Input_Evidence: <The Hunger Games is a 2012 American dystopian science fiction-adventure film directed by Gary Ross and is inspired by "Theseus and the Minotaur.">
                VERDICT: [SUPPORT]""",
                """Input_Claim: <People who are born with red hair are more likely to be left-handed.> Input_Evidence: <Red hair, also known as orange hair or ginger hair, is a human hair color found in 1–2% of the world population, appearing with greater frequency (2–6%) among people of Northern or Northwestern European ancestry and lesser frequency in other populations.>
                VERDICT: [NOT_ENOUGH_INFO]"""]
            if factuality:
                self.prompt_inst = "Can we infer the following statement from the given document? ONLY output the following options based on the given INPUT_Document and the Input_Statement: [Yes or No]"

                few_shot_prompt = [""" Input_Document: <The Hunger Games is a 2012 American dystopian science fiction-adventure film directed by Gary Ross and is inspired by "Theseus and the Minotaur."> Input_Statement: <Hunger Games is a movie inspired by the Greek myth.> OUTPUT: [Yes]
                Input_Document: <The Hunger Games is a 2012 American dystopian science fiction-adventure film directed by Gary Ross and is inspired by "Theseus and the Minotaur."> Input_Statement: <Hunger Games is a sci-fi movie inspired by Interstellar.> OUTPUT: [No]"""]
            
            
            if not nei:
                few_shot_prompt = few_shot_prompt[:-1]
            random.shuffle(few_shot_prompt)
            if few_shot:
                self.prompt_inst = self.prompt_inst +"\n".join(few_shot_prompt)

            


class CoTPrompting():
    def __init__(self,overal_inst,few_shot=True,nei = False,factuality = False):
        super().__init__()
        self.overal_inst = overal_inst
        if nei:
            self.prompt_inst = "First think step by step. Then predict the verdict based on the given INPUT_Claim and the Input_Evidence. Choose the output from the fllowing options: CONTRADICT, NOT_ENOUGH_INFO, SUPPORT. The output format: EXPLANATION: [rationale] VERDICT: [verdict]" 
        else:
            self.prompt_inst = "First think step by step. Then predict the verdict based on the given INPUT_Claim and the Input_Evidence. Choose the output from the fllowing options: CONTRADICT, SUPPORT. The output format: EXPLANATION: [rationale] VERDICT: [verdict]"
        few_shot_prompt =["""Input_Claim: <12 percent of the population in the US is suffering from diabetes.> Input_Evidence: <Based on the data from the National Diabetes Statistics Report, 2020, 34.2 million Americans, or 10.5% of the population, had diabetes. Approximately 1.6 million Americans have type 1 diabetes.>
        Explanation: [The evidence states that 10.5% of the population in the US is suffering from diabetes. This is less than 12 percent of the population in the US. Therefore, the claim is not supported by the evidence.] VERDICT: [CONTRADICT]""",
        """Input_Claim: <Rick and Morty is animated series for children and created by Dan Roiland.> Input_Evidence: <Rick and Morty is an American adult animated science fiction sitcom created by Justin Roiland and Dan Harmon for Cartoon Network's nighttime programming block Adult Swim.>
        Explanation: [The evidence states that Rick and Morty is created by Justin Roiland and Dan Harmon for Cartoon Network's nighttime programming block Adult Swim. This is not the same as the claim which states Rick and Morty is created by Dan Roiland. Therefore, the claim is not supported by the evidence.] VERDICT: [CONTRADICT]""",
        """Input_Claim: <Covid-19 is a virus that is transmitted through the air.> Input_Evidence: <COVID-19 is a respiratory disease caused by the SARS-CoV-2 virus. It is primarily spread through respiratory droplets when an infected person coughs, sneezes, or talks.>
        Explanation: [The evidence states that Covid-19 is primarily spread through respiratory droplets when an infected person coughs, sneezes, or talks. This is the same as the claim which states Covid-19 is a virus that is transmitted through the air. Therefore, the claim is supported by the evidence.] VERDICT: [SUPPORT]""",
        """Input_Claim: <Hunger Games is a movie inspired by the Greek myth. > Input_Evidence: <The Hunger Games is a 2012 American dystopian science fiction-adventure film directed by Gary Ross and is inspired by "Theseus and the Minotaur.">
        Explanation: [The evidence states that The Hunger Games is inspired by "Theseus and the Minotaur." This is the same as the claim which states Hunger Games is a movie inspired by the Greek myth. Therefore, the claim is supported by the evidence.] VERDICT: [SUPPORT]""",
        """Input_Claim: <People who are born with red hair are more likely to be left-handed.> Input_Evidence: <Red hair, also known as orange hair or ginger hair, is a human hair color found in 1–2% of the world population, appearing with greater frequency (2–6%) among people of Northern or Northwestern European ancestry and lesser frequency in other populations.>
        Explanation: [The evidence states that Red hair, also known as orange hair or ginger hair, is a human hair color found in 1–2% of the world population, appearing with greater frequency (2–6%) among people of Northern or Northwestern European ancestry and lesser frequency in other populations. The evidence does not provide enough information to support or contradict the claim.] VERDICT: [NOT_ENOUGH_INFO]"""]
        if not nei:
            few_shot_prompt = few_shot_prompt[:-1]
        if factuality:
                self.prompt_inst = "Can we infer the following statement from the given document? First think step by step and then output the following options based on the given INPUT_Document and the Input_Statement: [Yes or No]. Output format: EXPLANATION: [rationale] OUTPUT: [Yes or No]"

                few_shot_prompt = [""" Input_Document: <The Hunger Games is a 2012 American dystopian science fiction-adventure film directed by Gary Ross and is inspired by "Theseus and the Minotaur."> Input_Statement: <Hunger Games is a movie inspired by the Greek myth.> Explanation: [Input statement is aligned with the input document and can be inferred from the document.] OUTPUT: [Yes]
                Input_Document: <The Hunger Games is a 2012 American dystopian science fiction-adventure film directed by Gary Ross and is inspired by "Theseus and the Minotaur."> Input_Statement: <Hunger Games is a sci-fi movie inspired by Interstellar.> Explanation: [The input statement is not aligned with the input document and cannot be inferred from the document.] OUTPUT: [No]"""]

        # few_shot_prompt = few_shot_prompt_scifact_10_cot
        random.shuffle(few_shot_prompt)
        self.prompt_inst = self.overal_inst + self.prompt_inst
        if few_shot:
            self.prompt_inst = self.prompt_inst +"\n".join(few_shot_prompt)




class CARPPrompting():
    def __init__(self,overal_inst):
        super().__init__()
        self.overal_inst = overal_inst
        self.prompt_inst = self.overal_inst+ """First, present CLUES (i.e., keywords, phrases, contextual information, semantic relations, semantic meaning,
            tones, references) that help predict the verdict between INPUT_CLaim and INPUT_Evidence.
            Second, deduce a diagnostic REASONING process from premises (i.e., clues, INPUT_CLaim and INPUT_Evidence) that help verify/unverify the Input_Claim (Limit the number of words to 100).
            Third, PREDICT the overall verdict of INPUT_Claim with respect to Input_Evidence considering CLUES, and the REASONING. To predict VERDICT use following choices: [SUPPORT, CONTRADICT, NOT_ENOUGH_INFO].
            The output format:
            CLUES: []
            REASONING: []
            VERDICT: []""" 

class FactDetectPrompting(): 
    def __init__(self,overal_inst,few_shot=True,nei = False,factuality = False):
        super().__init__()
        self.overal_inst = overal_inst
        few_shot_prompt =["""Input_Claim: <12 percent of the population in the US is suffering from diabetes.> Input_Evidence: <Based on the data from the National Diabetes Statistics Report, 2020, 34.2 million Americans, or 10.5% of the population, had diabetes. Approximately 1.6 million Americans have type 1 diabetes.> Input_Facts: <[34.2 million Americans, or 10.5% of the population, had diabetes., Approximately 1.6 million Americans have type 1 diabetes.]>
        Relevant_Facts: [34.2 million Americans, or 10.5% of the population had diabetes.] Explanation: [The Input_Claim states that 12 percent of population in the US is suffering from diabetes while in the evidence it states 10.5% of americans have diabetes which is a contradiction.] VERDICT: [CONTRADICT]""",
        """Input_Claim: <Rick and Morty is animated series for children and created by Dan Roiland.> Input_Evidence: <Rick and Morty is an American adult animated science fiction sitcom created by Justin Roiland and Dan Harmon for Cartoon Network's nighttime programming block Adult Swim.> Input_Facts: <[Rick and Morty is an American adult animated science fiction sitcom., Rick and Morty is created by Justin Roiland and Dan Harmon., Rick and Morty is created by Justin Roiland and Dan Harmon for Cartoon Network's nighttime programming block Adult Swim]>
        Relevant_Facts: [Rick and Morty is created by Justin Roiland and Dan Harmon for Cartoon Network's nighttime programming block Adult Swim.] Explanation: [input claim states Rick and Morty is created by Dan Roiland and is created for children whereas evidence states that Rick and Morty is created by Justin Roiland and Dan Harmon and is created for adults which is a contradiction.] VERDICT: [CONTRADICT]""",
        """Input_Claim: <Covid-19 is a virus that is transmitted through the air.> Input_Evidence: <COVID-19 is a respiratory disease caused by the SARS-CoV-2 virus. It is primarily spread through respiratory droplets when an infected person coughs, sneezes, or talks.> Input_Facts: <[COVID-19 is a respiratory disease caused by the SARS-CoV-2 virus., Covid-19 is primarily spread through respiratory droplets when an infected person coughs, sneezes, or talks.]>
        Relevant_Facts: [Covid-19 is primarily spread through respiratory droplets when an infected person coughs, sneezes, or talks.] Explanation: [Claim states that covid-19 spreads through air and  evidence provides a related supporting fact for input claim which states Covid-19 is a virus that is transmitted through respiratory droplets.] VERDICT: [SUPPORT]""",
        """Input_Claim: <Hunger Games is a movie inspired by the Greek myth. > Input_Evidence: <The Hunger Games is a 2012 American dystopian science fiction-adventure film directed by Gary Ross and is inspired by "Theseus and the Minotaur."> Input_Facts: <[The Hunger Games is a 2012 American dystopian science fiction-adventure film directed by Gary Ross., The Hunger Games is inspired by "Theseus and the Minotaur."]>
        Relevant_Facts: [The Hunger Games is inspired by "Theseus and the Minotaur."] Explanation: [claim states that Hunger Games is inspired by Greek myth and evidence provides a related supporting fact for input claim which states Hunger Games is a movie inspired by "Theseus and the Minotaur" which is a Greek myth.] VERDICT: [SUPPORT]""",
        """Input_Claim: <People who are born with red hair are more likely to be left-handed.> Input_Evidence: <Red hair, also known as orange hair or ginger hair, is a human hair color found in 1–2% of the world population, appearing with greater frequency (2–6%) among people of Northern or Northwestern European ancestry and lesser frequency in other populations.> Input_Facts: <[Red hair, also known as orange hair or ginger hair, is a human hair color found in 1–2% of the world population., Red hair, also known as orange hair or ginger hair, is a human hair color found in 1–2% of the world population, appearing with greater frequency (2–6%) among people of Northern or Northwestern European ancestry and lesser frequency in other populations.]>
        Relevant_Facts: [] Explanation: [Evidence does not provide a relevant information to predict claim. Claim states that people with red hair are left-handed however, the evidence does not mention anything about left-handedness of red hair people.] VERDICT: [NOT_ENOUGH_INFO]"""]
        # few_shot_prompt =["""Input_Claim: <12 percent of the population in the US is suffering from diabetes.> Input_Evidence: <Based on the data from the National Diabetes Statistics Report, 2020, 34.2 million Americans, or 10.5% of the population, had diabetes. Approximately 1.6 million Americans have type 1 diabetes.> Important_Evidence_Facts: [34.2 million Americans, or 10.5% of the population had diabetes] 
        # Explanation on Importance: [This important fact from evindence provides a related counter-fact for input claim which states 12 percent of population in US have diabetes.] VERDICT: [CONTRADICT]""",
        # """Input_Claim: <Rick and Morty is animated series for children and created by Dan Roiland.> Input_Evidence: <Rick and Morty is an American adult animated science fiction sitcom created by Justin Roiland and Dan Harmon for Cartoon Network's nighttime programming block Adult Swim.> Important Facts: [Rick and Morty is created by Justin Roiland and Dan Harmon for Cartoon Network's nighttime programming block Adult Swim.] 
        # Explanation on Importance: [This important fact helps to reject claim because it provides a related counter-fact for input claim which states Rick and Morty is created by Justin Roiland and Dan Harmon not Dan Roiland and is created for adults not children.] VERDICT: [CONTRADICT]""",
        # """Input_Claim: <Covid-19 is a virus that is transmitted through the air.> Input_Evidence: <COVID-19 is a respiratory disease caused by the SARS-CoV-2 virus. It is primarily spread through respiratory droplets when an infected person coughs, sneezes, or talks.> Important_Evidence_Facts: [Covid-19 is primarily spread through respiratory droplets when an infected person coughs, sneezes, or talks.] 
        # Explanation on Importance: [This important fact is crucial to verify claim because it provides a related supporting fact for input claim which states Covid-19 is a virus that is transmitted through the air.] VERDICT: [SUPPORT]""",
        # """Input_Claim: <Hunger Games is a movie inspired by the Greek myth. > Input_Evidence: <The Hunger Games is a 2012 American dystopian science fiction-adventure film directed by Gary Ross and is inspired by "Theseus and the Minotaur."> Important_Evidence_Facts: [The Hunger Games is inspired by "Theseus and the Minotaur."] 
        # Explanation on Importance: [This fact important to verify claim because it provides a related supporting fact for input claim which states Hunger Games is a movie inspired by the Greek myth.] VERDICT: [SUPPORT]""",
        # """Input_Claim: <People who are born with red hair are more likely to be left-handed.> Input_Evidence: <Red hair, also known as orange hair or ginger hair, is a human hair color found in 1–2% of the world population, appearing with greater frequency (2–6%) among people of Northern or Northwestern European ancestry and lesser frequency in other populations.> Important_Evidence_Facts: [] 
        # Explanation on Importance: [there are no important facts related to claim in the evidence. Therefore, the evidence does not provide enough information to predict claim.] VERDICT: [NOT_ENOUGH_INFO]"""]
        # randomly shuffle the few shot prompts
        if not nei:
            few_shot_prompt = few_shot_prompt[:-1]
        # few_shot_prompt = few_shot_prompt_scifact_10
        random.shuffle(few_shot_prompt)
        if nei:
            self.instructions = """The input is the following format: Input_Claim: <>, Input_Evidence: <>, Input_Facts : <> First determine which of the facts from the Input_Facts list (seprated with ,) is related to Input_Claim. Then PREDICT the overall verdict of Input_Claim with respect to Input_Evidence. To predict VERDICT use following choices: [SUPPORT, CONTRADICT, NOT_ENOUGH_INFO].
            Output format: Relevant_Facts: [] Explanation: []  VERDICT: [] """
        else:
            self.instructions = """The input is the following format: Input_Claim: <>, Input_Evidence: <>, Input_Facts : <> First determine which of the facts from the Input_Facts list (seprated with ,) is related to Input_Claim. Then PREDICT the overall verdict of Input_Claim with respect to Input_Evidence. To predict VERDICT use following choices: [SUPPORT, CONTRADICT].
            Output format: Relevant_Facts: [] Explanation: []  VERDICT: [] """

        # if nei:
        #     self.instructions = """The input is the following format: Input_Claim: <>, Input_Evidence: <>, Important_Facts : <> Given important facts make a connection between these and the Input_Claim and Input_Evidence to PREDICT the overall verdict of Input_Claim with respect to Input_Evidence. To predict VERDICT use following choices: [SUPPORT, CONTRADICT, NOT_ENOUGH_INFO].
        #     Output format: Explanation: []  VERDICT: [] """
        # else:
        #     self.instructions = """The input is the following format: Input_Claim: <>, Input_Evidence: <>, Important_Facts : <> Given important facts make a connection between these and the Input_Claim and Input_Evidence to PREDICT the overall verdict of Input_Claim with respect to Input_Evidence. To predict VERDICT use following choices: [SUPPORT, CONTRADICT].
        #     Output format: Explanation: []  VERDICT: [] """
        if factuality:
            
            self.instructions = """The input is the following format: Input_Document: <>, Input_Statement: <>,  Input_Facts : <> First, determine which of the facts from the Input_Facts list (seprated with ,) is related to Input_Statement AND Input_Document. Then PREDICT whether the Input_Statement can be inferred from the Input_Document. To predict OUTPUT use following choices: [Yes, No].
            Output format: Relevant_Facts: [] Explanation: []  OUTPUT: [] """
            few_shot_prompt = [""" Input_Document: <The Hunger Games is a 2012 American dystopian science fiction-adventure film directed by Gary Ross and is inspired by "Theseus and the Minotaur."> Input_Statement: <Hunger Games is a movie inspired by the Greek myth.> Input_Facts: <[The Hunger Games is inspired by "Theseus and the Minotaur."]> Relevant_Facts: [The Hunger Games is inspired by "Theseus and the Minotaur."] Explanation: [Based on the relevant facts and its alignment to the input document, the statement can be inferred from the document.] OUTPUT: [Yes]
            Input_Document: <The Hunger Games is a 2012 American dystopian science fiction-adventure film directed by Gary Ross and is inspired by "Theseus and the Minotaur."> Input_Statement: <Hunger Games is a sci-fi movie inspired by Interstellar.> Input_Facts: <[The Hunger Games is inspired by Interstellar,"The Hunger Games is a 2012 American dystopian science fiction-adventure film"]> Relevant_Facts: [The Hunger Games is a 2012 American dystopian science fiction-adventure film"] Explanation: [ The relevant facts do not provide enough information to infer the statement from the document.] OUTPUT: [No]"""]
        
        self.prompt_inst = self.overal_inst+self.instructions
        if few_shot:
            self.prompt_inst = self.prompt_inst +"\n".join(few_shot_prompt)
        # self.prompt_inst = self.overal_inst+ """ Look into the Input_Claim and generate the MOST IMPORTANT shorter sentences from the Input_Evidence as necessary CLUES to predict the verdict. Then reason over each CLUE to predict the verdict. To predict VERDICT use following choices: [SUPPORT, CONTRADICT, NOT_ENOUGH_INFO].
        # Output format:  CLUES: [] REASONING: [] VERDICT: [] """

    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






class FactDetectModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.model.config.pad_token_id = self.model.config.eos_token_id
        self.loss_fct = nn.CrossEntropyLoss()
        self.metrics = {}
        metrics = {}
        fold_names = ["train", "valid", "test"]
        for name in fold_names:
            metrics[f"metrics_{name}"] = ClassificationMetric()

        self.metrics = nn.ModuleDict(metrics)
    def forward(self, input_ids, attention_mask):
        if len(input_ids.squeeze(1).size()) == 2:
            bsz, src_len = input_ids.squeeze(1).size()
        else:
            print(f"Expected mask to be a 2D tensor, but got a tensor with shape {input_ids.squeeze(1).size()}")
                
        
        outputs = self.model(input_ids = input_ids.squeeze(1), attention_mask=attention_mask.squeeze(1))
        return outputs.logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        logits = self(input_ids, attention_mask)

        #weighted loss based on the weight of each class
        label_loss = F.cross_entropy(
            logits, labels, reduction="none")
        # Take weighted average of per-sample losses.
        label_loss = (batch["weight"] * label_loss).sum()
        loss = label_loss

        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        logits = self(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1).detach()
        self._invoke_metrics(preds, batch, "valid")
        self._log_metrics("train", on_epoch=True)
        self._log_metrics("valid", on_epoch=True)
        # loss = self.loss_fct(logits.view(-1, 3), labels.view(-1))
        # self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    

    
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, logits = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)
    
    
    def predict(self,input_ids,attention_mask):
        
        with torch.no_grad():
            logits = self(input_ids, attention_mask)
        return logits
    
    def _invoke_metrics(self, pred, batch, fold):
        """
        Invoke metrics for a single step of train / validation / test.
        `batch` is gold, `pred` is prediction, `fold` specifies the fold.
        """
        assert fold in ["train", "valid", "test"]

        # We won't need gradients.
        # detached = {k: v.detach() for k, v in pred.items()}
        # Invoke the metrics appropriate for this fold.
        self.metrics[f"metrics_{fold}"](pred, batch)

    def _log_metrics(self, fold, on_epoch=True):
        "Log metrics for this epoch."
        the_metric = self.metrics[f"metrics_{fold}"]
        to_log = the_metric.compute()
        the_metric.reset()
        for k, v in to_log.items():
            self.log(f"{fold}_{k}", v, on_epoch=on_epoch)



