import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM


# Loading the Data
with open("train_qa.txt", "rb") as fp:   # Unpickling
    train_data = pickle.load(fp)
with open("test_qa.txt", "rb") as fp:   # Unpickling
    test_data =  pickle.load(fp)

print(len(test_data))
print(len(train_data))

print(' '.join(train_data[0][0]))


# Create a set that holds the vocab words
vocab = set()
all_data = test_data + train_data

for story, question, answer in all_data:
    # In case you don't know what a union of sets is:
    # https://www.programiz.com/python-programming/methods/set/union
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))
vocab.add('no')
vocab.add('yes')

print(vocab)
vocab_len = len(vocab) + 1 #we add an extra space to hold a 0 for Keras's pad_sequences

max_story_len = max([len(data[0]) for data in all_data])


max_question_len = max([len(data[1]) for data in all_data])

tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

print(tokenizer.word_index)

train_story_text = []
train_question_text = []
train_answers = []

for story,question,answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)
    train_answers.append(answer)

train_story_seq = tokenizer.texts_to_sequences(train_story_text)

def vectorize_stories(data, word_index=tokenizer.word_index,max_story_len = max_story_len, max_question_len=max_question_len):
    # X = STORIES
    X = []
    # Xq = QUERY/QUESTION
    Xq = []
    # Y = CORRECT ANSWER
    Y = []
    for story, query, answer in data:
        # Grab the word index for every word in story
        x = [word_index[word.lower()] for word in story]
        # Grab the word index for every word in query
        xq = [word_index[word.lower()] for word in query]

        # Grab the Answers (either Yes/No so we don't need to use list comprehension here)
        # Index 0 is reserved so we're going to use + 1
        y = np.zeros(len(word_index) + 1)

        # Now that y is all zeros and we know its just Yes/No , we can use numpy logic to create this assignment
        #
        y[word_index[answer]] = 1

        # Append each set of story,query, and answer to their respective holding lists
        X.append(x)
        Xq.append(xq)
        Y.append(y)

        # Finally, pad the sequences based on their max length so the RNN can be trained on uniformly long sequences.

        # RETURN TUPLE FOR UNPACKING
    return (pad_sequences(X, maxlen=max_story_len), pad_sequences(Xq, maxlen=max_question_len), np.array(Y))

inputs_train, queries_train, answers_train = vectorize_stories(train_data)
inputs_test, queries_test, answers_test = vectorize_stories(test_data)


input_sequence = Input((max_story_len,))
question_input = Input((max_question_len,))

vocab_size = len(vocab)+1


input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=64))
input_encoder_m.add(Dropout(0.3))

input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,output_dim=max_question_len))
input_encoder_c.add(Dropout(0.3))

question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=max_question_len))
question_encoder.add(Dropout(0.3))

input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question_input)

match = dot([input_encoded_m, question_encoded], axes=(2,2))
match = Activation('softmax')(match)

response = add([match, input_encoded_c])
response = Permute((2,1))(response)

answer = concatenate([response, question_encoded])

answer = LSTM(32)(answer)

answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer)
answer = Activation('softmax')(answer)

model = Model([input_sequence, question_input], answer)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])


history = model.fit([inputs_train, queries_train], answers_train,batch_size=32,epochs=120,validation_data=([inputs_test, queries_test], answers_test))

import matplotlib.pyplot as plt
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


pred_results = model.predict(([inputs_test, queries_test]))

story =' '.join(word for word in test_data[0][0])
print(story)


query = ' '.join(word for word in test_data[0][1])
print(query)


print("True Test Answer from Data is:",test_data[0][2])

print(pred_results[0])



val_max = np.argmax(pred_results[0])
print(val_max)

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key

print("Predicted answer is: ", k)
print("Probability of certainty was: ", pred_results[0][val_max])


my_story = "John left the kitchen . Sandra dropped the football in the garden ."
my_story.split()

my_question = "Is the football in the garden ?"

mydata = [(my_story.split(),my_question.split(),'yes')]

my_story,my_ques,my_ans = vectorize_stories(mydata)

pred_results = model.predict(([ my_story, my_ques]))


val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key

print('my story')
print("Predicted answer is: ", k)
print("Probability of certainty was: ", pred_results[0][val_max])




