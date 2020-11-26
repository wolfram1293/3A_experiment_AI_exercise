import sys
import torch

import sentence_data
from sentence_data import UNKNOWN_WORD_ID
from translator_model import TranslatorModel

dataset = sentence_data.SentenceData("dataset/data_1000.txt")# 2.7.6ではdata_full.txtとする

model = TranslatorModel(dataset.english_word_size(),
                        dataset.japanese_word_size())

model.load_state_dict(torch.load("trained_model/translator_10.model"))# 2.7.6ではtranslator_full.modelとする

# 入力された文章を単語に分割する
sentence = input("input an english sentence : ").split(' ')
# 単語IDのリストに変換する
sentence_id = []
for word in sentence:
    if not word:
        # 単語が空だったら飛ばす
        continue
    word = word.lower()
    id = dataset.english_word_id(word)
    if id is None:
        #sys.stderr.write("Error : Unknown word " + word + "\n")
        #sys.exit()
        id = torch.tensor(UNKNOWN_WORD_ID,dtype=torch.long).unsqueeze(-1)
        sentence_id.append(id)
    else:
        id = torch.tensor(id,dtype=torch.long).unsqueeze(-1)
        sentence_id.append(id)
japanese = model(torch.stack(sentence_id))
for id in japanese:
    print(dataset.japanese_word(id), end='')
print()
