import json
import time
import string
import re
import os
from IPython.utils import io
import ollama

from answerfromweb import get_answer

class FactCheckingLLM:
    def __init__(self):
        
        ollama.chat(model=self.model, messages=[{"role": "user", "content": self.prompt[0]}])
    model = "llama3:latest"
    example_claims = [
    "Emerson Moser, who was Crayola’s top crayon molder for almost 40 years, was colorblind.",
    "Bernie Sanders said 85 million Americans have no health insurance.",
    "JAG charges Nancy Pelosi with treason and seditious conspiracy.",
    "Cheri Beasley “backs tax hikes — even on families making under $75,000.",
    # Add more example claims as needed
    ]
    prompt = ['''
          
Claim: "Emerson Moser, who was Crayola’s top crayon molder for almost 40 years, was colorblind."
A fact checker will decompose the claim into 4 subclaims that are easier to verify:
1.Emerson Moser was a crayon molder at Crayola.
2.Moser worked at Crayola for almost 40 years.
3.Moser was Crayola's top crayon molder.
4.Moser was colorblind.
To verify subclaim 1, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Is there any official record or documentation indicating that Emerson Moser worked as a crayon molder at Crayola?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: Yes.
To verify subclaim 2, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Are there any official records or documentation confirming Emerson Moser's length of employment at Crayola?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: Emerson Moser, who is retiring next week after 35 years, isn't colorblind in the sense that he can't see color at all. It's just that some ...
To verify subclaim 3, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Are there credible sources or publications that mention Emerson Moser as Crayola's top crayon molder?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: Yes.
To verify subclaim 4, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Are there any credible sources or records indicating that Emerson Moser was colorblind?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: Yes.
Question: Was Emerson Moser's colorblindness only confusing for certain colors?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: Moser has had tritanomaly, a type of colorblindness that makes it difficult to distinguish between blue and green and between yellow and red.
Based on the answers to these questions, it is clear that among pants-fire, false, barely-true, half-true, mostly-true, and true, the claim can be classified as mostly-true.

Claim: "Bernie Sanders said 85 million Americans have no health insurance."
A fact checker will not split the claim since the original claim is easier to verify.
To verify the claim, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: How many Americans did Bernie Sanders claim had no health insurance?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: "We have 85 million Americans who have no health insurance," Sanders said Dec. 11 on CNN's State of the Union.
Question: How did Bernie Sanders define "no health insurance"?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: Sanders spokesperson Mike Casca said the senator was referring to the number of uninsured and under-insured Americans and cited a report about those numbers for adults.
Question: How many Americans were uninsured or under-insured according to the Commonwealth Fund survey?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: The Commonwealth Fund survey found that 43% of working-age adults 19 to 64, or about 85 million Americans, were uninsured or inadequately insured.
Question: Is the statement "we have 85 million Americans who have no health insurance" partially accurate according to the information in the passage?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: Bernie Sanders omitted that his figure included people who either have no health insurance or are under-insured.
Based on the answers to these questions, it is clear that among pants-fire, false, barely-true, half-true, mostly-true, and true, the claim is classified as half-true.

Claim: "JAG charges Nancy Pelosi with treason and seditious conspiracy."
A fact checker will decompose the claim into 2 subclaims that are easier to verify:
1. JAG has made a claim or accusation against Nancy Pelosi.
2. The specific charges or allegations made against Nancy Pelosi are treason and seditious conspiracy.
To verify subclaim 1, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Is it true that JAG has made a claim or accusation against Nancy Pelosi?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: There is no evidence to support this claim and a spokesperson for the U.S. Navy Judge Advocate General's Corps has stated that it is not true.
To verify subclaim 2, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Is it true that the specific charges or allegations made against Nancy Pelosi are treason and seditious conspiracy?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: There is no evidence to support this claim.
Question: Where is the source of the claim?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: Real Raw News, a disclaimer stating that it contains "humor, parody and satire" and has a history of publishing fictitious stories.
Based on the answers to these questions, it is clear that among pants-fire, false, barely-true, half-true, mostly-true, and true, the claim is classified as pants-fire.

Claim: "Cheri Beasley “backs tax hikes — even on families making under $75,000."
A fact checker will decompose the claim into 2 subclaims that are easier to verify:
1.Cheri Beasley supports tax increases.
2.Cheri Beasley supports tax increases for families with an income under $75,000.
To verify subclaim 1, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she might raise each question and look for an answer:
Question: Does Cheri Beasley support tax increases?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: Beasley supports student loan bailouts for the wealthy.
To verify subclaim 2, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she might raise each question and look for an answer:
Question: Does the ad accurately link Beasley's position on student loan debt forgiveness with her stance on tax hikes for families making under $75,000 per year?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: The ad makes a misleading connection between the two issues and does not accurately represent Beasley's position on tax hikes for families making under $75,000 per year.
Question: Has Cheri Beasley ever advocated for tax hikes specifically on families making under $75,000?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: No evidence found that Cheri Beasley has explicitly advocated for such a tax hike.
Based on the answers to these questions, it is clear that among pants-fire, false, barely-true, half-true, mostly-true, and true, the claim can be classified as barely-true.


Claim: ''', '''A fact checker will''',
 ]

    def promptf(self,question, prompt, intermediate="\nAnswer:", followup="Intermediate Question", finalans='\nBased on the answers to these questions, it is clear that among pants-fire, false, barely-true, half-true, mostly-true, and true, the claim '):
        cur_prompt =question + prompt[1]
        if question in self.example_claims:
            pass
        else:
            ret_text = self.chat_with_ollama(cur_prompt, stop='Answer me ‘yes’ or ‘no’: No.')
            while True:
                cur_prompt += ret_text + 'Answer me ‘yes’ or ‘no’: No.'
                question = ret_text.split('\nTell me')[0].split('\n')[-1]
                question = self.extract_question(ret_text)
                print('question')
                print(question)
                print('Answer:')
                external_answer = get_answer(question)
                cur_prompt += intermediate + ' ' + external_answer + '.\n' 
                ret_text = self.chat_with_ollama(cur_prompt, 'Answer me ‘yes’ or ‘no’: No.')
                break

        cur_prompt += finalans
        cur_prompt += 'can be classified as'
        ret_text = self.chat_with_ollama(cur_prompt, '\n')

        return cur_prompt + ret_text
    def chat_with_ollama(self,prompt, stop=["\n"]):
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        assistant_message = response['message']['content']
    
        return assistant_message

    def extract_question(self,generated):
        # Extract the question more robustly
        question_part = generated.split('Question: ')[-1].split('Answer')[0].strip()
        return question_part if question_part else "What is the verification status?"

factchecker=FactCheckingLLM()
json_list=[{"label":"questiob","claim":"cookies cause cancer","event_id":"1234"}]
for json_str in json_list:
        result = json_str
        label = result["label"]
        claim = result["claim"]
        idx = result["event_id"]

        print(str(idx)+'\n')
        question = claim 
        ans = factchecker.promptf(question, factchecker.prompt)


        print('\nresult')
        print(ans)
        print('label')
        print(label)


