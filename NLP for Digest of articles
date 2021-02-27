import sys
import numpy as np
import re
#Библиотеки для стоп-слов
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=stopwords.words("russian")
#Библиотеки для нормализации
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
import math

with open("Выбранный_вручную_текст_БЕЗ_55.txt", "r", encoding="utf-8") as text:
    text_str = text.read()

def delete_str_translation(string): # функция удаления знака переноса строки из всего текста
    string = string.replace('-\n\n', '')
    string = string.replace('-\n', '')
    string = string.replace('\n', ' ')
    string = string.replace('-', ' ')
    return string

text_string=delete_str_translation(text_str)

def tokenization_on_words(string):
    return re.findall(r'\b([-\w]+)\b', string)  # в строке text_str находит шаблон, прописанный слева

def Normalize_and_DeleteStopWords(array): # Формируем один массив всего текста в виде ['слово1_норм_ф', 'слово2_норм_ф','слово3_норм_ф',.....] где слова могут повторяться
    return [morph.parse(word)[0].normal_form for word in array if word not in stop_words and len(word)>=3]

def Delete_numbers(array):
    return [ex for ex in array if ex.isalpha()]

def Delete_double_word(array):
    return sorted(set(array), key=lambda x: array.index(x))

words_with_double=Delete_numbers(Normalize_and_DeleteStopWords(tokenization_on_words(text_string)))

#СОЗДАНИЕ МАССИВА ВСЕХ КОЛЛОКАЦИЙ
from nltk.collocations import *
N_best = 100 # number of bigrams to extract
# class for association measures
bm = nltk.collocations.BigramAssocMeasures()
# class for bigrams extraction and storing
f = BigramCollocationFinder.from_words(words_with_double)
# remove too seldom bigrams
f.apply_freq_filter(5)

# pmi_collocations = [' '.join(i) for i in f.nbest(bm.pmi, N_best)] # лист строчек коллокаций
pmi_collocations = f.nbest(bm.pmi, N_best) # лист тьюплов коллокаций

print(pmi_collocations)
#КОНЕЦ СОЗДАНИЯ МАССИВА ВСЕХ КОЛЛОКАЦИЙ

def tokenization_on_words_and_collocations(array_of_all_words):
    temporary_container=[]
    elements_of_text=[]
    for it in range(len(array_of_all_words)):
        if it==range(len(array_of_all_words)):
            temporary_container.append(array_of_all_words[it])
        else:
            if tuple(array_of_all_words[it:it+2]) in pmi_collocations:
                temporary_container.append(' '.join(array_of_all_words[it:it+2]))
            else:
                temporary_container.append(array_of_all_words[it])

    remove_index = None

    for token in temporary_container:
        if temporary_container.index(token)==remove_index:
            continue
        elif ' ' not in token:
            elements_of_text.append(token)
        elif ' ' in token:
            remove_index=temporary_container.index(token)+1
            elements_of_text.append(token)

    return elements_of_text

words_bigrams_with_double=tokenization_on_words_and_collocations(words_with_double) # слова и биграммы с повторениями

words_bigrams_not_double=Delete_double_word(words_bigrams_with_double) # слова и биграммы без повторений

# БЛОК 2
# РАБОТА С ЛИСТОМ ТЕКСТОВ

text_split = text_str.split('\n\n\n\n') # Формируем массив текстов, где на 0-м месте стоит первый текст, на 1 месте стоит второй текст и так далее

# формируем массив статей, для подсчета частоты слов (то есть есть повторяющиеся слова)
text_split_on_article=[tokenization_on_words_and_collocations(Delete_numbers(Normalize_and_DeleteStopWords(tokenization_on_words(delete_str_translation(text))))) for text in text_split]

# Создание словаря: слово в статье-частота встречи в статье
frequency_in_article = [{} for i in range(0, len(text_split_on_article))]
for i in range(0, len(text_split_on_article)):
    for word in words_bigrams_not_double:
        count = text_split_on_article[i].count(word)
        frequency_in_article[i][word] =count

list_of_tuple=nltk.FreqDist(words_bigrams_with_double).most_common()# Создание словаря: слово и частота встречи в рамках всего сборника Строчка 1 (из 1)

freq_in_full_corpus_dict={k:v for k,v in list_of_tuple} # Создание словаря: слово и частота встречи в рамках всего сборника Строчка 2 (из 2)

freq_in_full_corpus_no_twice_dict={k:v for k,v in freq_in_full_corpus_dict.items() if v>2} #Слова которые встречаются более одного раза в рамках всего сборника (Мешок слов)
                                                                                          # (Слово-частота встречи в корпусе)
freq_in_article_no_twice_dict=[{k:article[k] for k,_ in freq_in_full_corpus_no_twice_dict.items()} for article in frequency_in_article] #Слова которые встречаются более одного раза
                                                                                                                                      # в рамках всего сборника (Мешок слов)
                                                                                                                                      # (Слово-частота встречи в одной статье)

num_doc = []#Вектор: в скольких документах встечается данное слово (для Tf-IDf)
num=0
for word in list(freq_in_full_corpus_no_twice_dict.keys()):
    for i in range(0,len(text_split_on_article)):
        if frequency_in_article[i][word]>0:
            num+=1
    num_doc.append(num)
    num=0

how_many_documents_contains_word=dict(zip(list(freq_in_full_corpus_no_twice_dict.keys()),num_doc))

def Tf_Idf(array_of_dict):
    return [{k:round(v*math.log10((len(array_of_dict)/how_many_documents_contains_word[k])),3) for k,v in article.items()} for article in array_of_dict]

print(Tf_Idf(freq_in_article_no_twice_dict)[0])
print('V(сколько раз встечается в статье):',frequency_in_article[0]['сопоставимый'])
print('V(сколько раз встечается в статье), проверочный:',freq_in_article_no_twice_dict[0]['сопоставимый'])
print('Знаменатель (в скольких статьях встречается):',how_many_documents_contains_word['сопоставимый'])
print('Частота во всем корпусе:',freq_in_full_corpus_dict['сопоставимый'])
# sys.exit(1)
TF_IDF_list_of_dict=Tf_Idf(freq_in_article_no_twice_dict)

# print(len(list(TF_IDF_list_of_dict[0].keys())))
# print(TF_IDF_list_of_dict[1].values())

sys.exit(1) # перезапись с коллокациями уже была

with open('TF_IDF.txt', 'w') as testfile:
    for article in TF_IDF_list_of_dict:
        testfile.write(' '.join([str(a) for a in article.values()]) + '\n\n\n')

with open('Мешок_слов.txt', 'w', encoding="utf-8") as file:
    file.write('\n'.join(TF_IDF_list_of_dict[0].keys()))








