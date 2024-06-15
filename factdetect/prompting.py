from args import get_args
from models import Prompting,VanillaPrompting,CoTPrompting,CARPPrompting,FactDetectPrompting
import jsonlines
import torch
import tqdm as tqdm
import random
import json
from sklearn.metrics import f1_score, precision_score, recall_score,confusion_matrix
from sklearn.metrics import roc_auc_score
import pandas as pd 



class SciFactLabelPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, corpus: str, claims: str,af_mode = False,with_nei=True,factuality = False):
        self.samples = []
  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}

        for claim in jsonlines.open(claims):
           
                 
            if claim['evidence']:
                if factuality:
                    evidence = claim['evidence']
                    label = "Yes" if claim['label']=='0' else "No"
                    af_sentences = claim['af_sentences']
                    # af_ets = claim['af_sets']
                    if len(af_sentences) == 0:
                        af_sentences = []
                    entry = {
                        'claim': claim['claim'],
                        'rationale': evidence,
                        'label': label,
                        'af_sentences':af_sentences,
                        # 'af_sets':af_sets
                    }
                    self.samples.append(entry)
                else:

                    for doc_id, evidence_sets in claim['evidence'].items():
                        doc = corpus[int(doc_id)]

                        # Add individual evidence set as samples:
                        rationales = []
                        if isinstance(evidence_sets, list):
                            for evidence_set in evidence_sets:
                                rationales.extend([i for i in evidence_set['sentences']])
                        else:
                            rationales = [i for i in evidence_sets['sentences']]

                        #for scifact open 
                        
                        # rationales = evidence_sets['sentences']

                        rationale_sentences = [doc['abstract'][i].strip() for i in rationales]
                        if isinstance(evidence_sets, list):
                            gold_label = evidence_sets[0]['label']
                        else:
                            gold_label = evidence_sets['label']
                        entry = {
                            'claim': claim['claim'],
                            'rationale': ' '.join(rationale_sentences),
                            # 'label': evidence_sets['label'],
                            'label': gold_label,
                            'id':str(claim['id'])
            
                        }
                        if af_mode:
                            
                            af_sentences = claim['af_sentences'][doc_id]
                            af_sets = claim['af_sets'][doc_id]
                            if len(af_sentences) == 0:
                                if len(rationale_sentences) > 0:
                                    af_sentences = [rationale_sentences[0]]
                                else:
                                    af_sentences = ['']
                                
                            
                            entry.update({'af_sentences':af_sentences,
                                        'af_sets':af_sets})
                        self.samples.append(entry)

                        # print("claim",claim['claim'])
                        # print("rationale",' '.join(rationale_sentences))
                        # print("label",evidence_set['label'])
                        # print("=======================================")
                        
                        # 'af_sets':af_sets})
        

                        
                
            else:
                if not with_nei:
                    continue
                # Add negative samples
                if 'cited_doc_ids' not in claim:
                    doc_id = claim['doc_ids'][0]
                else:
                    doc_id =  claim['cited_doc_ids'][0]
                # doc_id = claim['doc_ids'][0]
                doc = corpus[int(doc_id)]
                min_len = min(len(doc['abstract']), 3)
                non_rationale_idx = random.sample(range(len(doc['abstract'])), k=random.randint(1, min_len))
                non_rationale_sentences = [doc['abstract'][i].strip() for i in non_rationale_idx]
            
                entry = {
                    'claim': claim['claim'],
                    'rationale': ' '.join(non_rationale_sentences),
                    'label': 'NOT_ENOUGH_INFO',
                    'id':str(claim['id'])
        
                }
                if af_mode:
                    af_sentences = [non_rationale_sentences[0]]
                    af_sets = []
                    entry.update({'af_sentences':af_sentences,
                                    'af_sets':af_sets
                                })
                self.samples.append(entry)
                # print("claim",claim['claim'])
                # print("rationale",' '.join(non_rationale_sentences))
                # print("label",'NOT_ENOUGH_INFO')
                # print("=======================================")
            
                # 'af_sets':af_sets})

            # print(entry)



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]






def write_inferences2file(args, inferences, out_file):
    with jsonlines.open(out_file, mode='w') as writer:
        for inference in inferences:
            writer.write(inference)



def get_inferences(args, model, dataset):
    inferences = []
    if args.llm_model_name == 'llama' or args.llm_model_name == 'mistral' or args.llm_model_name == 'vicuna':
        inferences = model.get_all_inferences(dataset)
        return inferences

    for data in tqdm.tqdm(dataset):
        claim, evidence, label,af_sentences,af_sets = data['claim'], data['rationale'], data['label'],data['af_sentences'],None
        inference = model.get_inference(claim,evidence,af_sentences,af_sets=af_sets)
        inference.update({
        'label':label,
        'claim':claim,
        'evidence':evidence})
        inferences.append(inference)
       

    return inferences


def main():
    args = get_args()

    # if args.mode =='train':

    model = Prompting(args,args.llm_model_name,nei=args.with_nei)
    
    dataset = SciFactLabelPredictionDataset(args.corpus, args.test_file,af_mode=args.af_mode,with_nei=args.with_nei,factuality = args.factuality).samples

    
    inferences = get_inferences(args, model, dataset)
    write_inferences2file(args, inferences, args.outfilename)


  



   


if __name__ == "__main__":
    main()
