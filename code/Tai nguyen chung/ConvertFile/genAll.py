import benepar, spacy
benepar.download('benepar_en3')
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
nltk.download('omw-1.4')
nltk.download('wordnet')
import re


from bs4 import BeautifulSoup 
import xml

# Reading the data inside the xml file to a variable under the name  data

import json


import os
pathData = "./data/SELECT"
listfile = os.listdir(pathData)


from collections import Counter

set_argument = set()

special_case = {
    "modify": ["modifing"],
    "abolish":["abolised"],
    "catalyse":["catalyze","catalyzing","catalyzed"]
}

count_train = 0
count_test = 0
count_dev = 0
count_write= 0
count_skip = 0
special_char = "@"



def find_sub_list1(sl,l):
    i = 0
    j = 0
    for i in range(len(l)):
        for j in range(len(sl)):
            # if (l[i].similarity(sl[j]) != 1.0):
            if(l[i] != sl[j]):
                break
            if(j == len(sl)-1):
                return (i-j,i)
            i+=1
            if (i > len(l)-1):
                return None
    return None

def tokenizer(s):
    b = re.split('(\W)', s)
    return [i for i in b if i!= " " and i !=""]
# def tokenizer(s):
    
#     return s.split()

set_argument = set()
with open("./error.txt","w") as err:
    with open("./train.tsv","w") as rs1:
        with open("./dev.tsv","w") as dev1:
            with open("./test.tsv","w") as test1:
                for file in listfile:
                    name1 = file.split('.')[0] #get name file in name.xml
                    name1 = name1.split('_')[0] #get verb in case a_1.xml
                    print(name1)
                    with open(f'{pathData}/{file}', 'r') as f:
                        xml_file = f.read()
                    bs_data = BeautifulSoup(xml_file, 'xml')
                    examples = bs_data.find_all('example') 
                    print(len(examples))
                    count = 0

                    for s in range(len(examples)):
                        temp_dict = {}
                        # check if example only have 1 sentence
                        sentence = examples[s].find('text').text.lower()
                        sentence = sentence.replace(".",special_char)
                        if sentence[-1:] == special_char:
                            sentence = sentence[:-1] + "."
                        else:
                            sentence = sentence + "."
                        
                        sentence_list = tokenizer(sentence)
                        # sentence_list_syntax = list(sentence_nlp.sents)
                        # sentence_syntax = sentence_list_syntax[0]._.parse_string
                        # xac dinh argument
                        # skip = False
                        for arg in examples[s].findAll('arg'):
                            n = 'A' + arg.get('n')
                            argument = arg.text.lower()
                            if len(argument) >0:
                                argument = argument.replace(".",special_char)
                                if(argument[-1:] == special_char):
                                    argument = argument[:-1]
                            # arg_list = list(nlp(argument))
                            arg_list = tokenizer(argument)
                            # arg_list = argument.split()
                            # arg_list = list(nlp(arg.text))
                            dict_value = find_sub_list1(arg_list,sentence_list)
                            if (dict_value != None):
                                temp_dict[n] = dict_value
                        word = []
                        label_BIO = ["O" for _ in range(len(sentence_list))]
                        is_write = False
                        real_root = []

                        for i in range(len(sentence_list)):
                            word.append(sentence_list[i])
                            
                            #normalizer and label verb
                            # WordNetLemmatizer().lemmatize(sentence_list[i],'v')
                            is_special = False
                            word_temp = WordNetLemmatizer().lemmatize(re.sub('\W+','', sentence_list[i]),'v')
                            try:
                                is_special = word_temp in special_case[name1]
                            except:
                                is_special = False

                            if(word_temp == name1 or is_special):
                                label_BIO[i]="V"
                                is_write = True
                        for k in temp_dict:
                            v = temp_dict.get(k)
                            if (v[0]!=v[1]):
                                label_BIO[v[0]]='B-'+k
                                for g in range(v[0]+1,v[1]+1):
                                    label_BIO[g] ="I-"+k
                            else:
                                label_BIO[v[0]]='B-'+k

                        if not is_write:
                            a = f"ERROR parsing lost preticate : {name1}\n sentence:\n{sentence_list}\n"
                                # is_write = False
                                # print("not write")
                            err.write(a)
                            err.write("\n")
                            count_skip+=1
                            print(a)
                            continue
                        count+=1
                        # print(f"count: {count}, file: {file_flag}")
                        for i in range(len(sentence_list)):
                            set_argument.add(label_BIO[i])

                            if(s <= len(examples)*2/3):
                                file_flag = 1
                                rs1.write(f"{word[i]}\t")
                                rs1.write(f"{label_BIO[i]}\n")                  
                            elif (s <= len(examples)*5/6):
                                file_flag = 2
                                dev1.write(f"{word[i]}\t") 
                                dev1.write(f"{label_BIO[i]}\n")
                            else:
                                file_flag = 3
                                test1.write(f"{word[i]}\t")
                                test1.write(f"{label_BIO[i]}\n")
                        count_write += 1
                        print(f"{name1}: {count}")
                        print(f"{s}/{len(examples)}")
                        print(f"count_write: {count_write}")
                        print(f"count_skip: {count_skip}")
                        if(file_flag == 1):
                            rs1.write(f"\n")
                        elif(file_flag == 2):
                            dev1.write(f"\n")
                        elif(file_flag==3):
                            test1.write(f"\n")
                        file_flag = 0
                        # break

                    # break


