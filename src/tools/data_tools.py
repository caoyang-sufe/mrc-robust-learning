# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
import os
import json
import pandas
from xml.etree import ElementTree


def yield_race_sample(race_path, types, difficulties, batch_size):
    current_batch_size = 0
    batch_data = list()
    for type_ in types:
        # train, dev, test
        for difficulty in difficulties:
            # high, middle
            data_root = os.path.join(race_path, type_, difficulty)
            for filename in os.listdir(data_root):
                with open(os.path.join(data_root, filename), 'r', encoding="utf8") as f:
                    data = json.load(f)
                article_id = data["id"]
                article = data["article"]
                questions = data["questions"]
                options = data["options"]
                answers = data["answers"]
                # Ensure the number of questions, options and answers are the same
                assert len(questions) == len(options) == len(answers), article_id
                # Ensure id matches filename
                assert article_id == difficulty + filename, article_id
                for question_id, (question, option, answer) in enumerate(zip(questions, options, answers)):
                    # Ensure total option is 4
                    assert len(option) == 4, f"{article_id}-{question_id}"
                    # @yield article_id	    : "high1.txt"
                    # @yield question_id	: 0
                    # @yield article		: "My husband is a born shopper. He ..."
                    # @yield question		: "The husband likes shopping because   _  ."
                    # @yield options	    : ["he has much money.", "he likes the shops.", "he likes to compare the prices between the same items.", "he has nothing to do but shopping."]
                    # @yield answer		    : 'C'
                    batch_data.append({"article_id": article_id,
                                       "question_id": question_id,
                                       "article": article,
                                       "question": question,
                                       "options": option,
                                       "answer": answer,
                                       })
                    current_batch_size += 1
                    if current_batch_size == batch_size:
                        yield batch_data
                        current_batch_size = 0
                        batch_data = list()
    if current_batch_size > 0:
        yield batch_data


def yield_dream_sample(dream_path, types, batch_size):
    current_batch_size = 0
    batch_data = list()
    for type_ in types:
        # train, dev, test
        with open(os.path.join(dream_path, f"{type_}.json"), 'r', encoding="utf8") as f:
            data = json.load(f)
        for article_sentences, questions, article_id in data:
            article = '\n'.join(article_sentences)
            for question_id, question_item in enumerate(questions):
                question = question_item["question"]
                options = question_item["choice"]
                flag = False
                assert len(options) == 3, f"{article_id}-{question_id}"
                for i, option in enumerate(options):
                    if option == question_item["answer"]:
                        assert not flag, f"{article_id}-{question_id}"
                        answer = "ABC"[i]
                        flag = True
                assert flag, f"{article_id}-{question_id}"
                # @yield article_id	    : "4-199"
                # @yield question_id	: 0
                # @yield article		: "W: The movie next Tuesday ..."
                # @yield question		: "What can we conclude about the movie?"
                # @yield options		: ["They want to buy the tickets for the movie.", "The tickets for the movie were sold.", "The movie will not be shown."]
                # @yield answer		    : 'C'
                batch_data.append({"article_id": article_id,
                                   "question_id": question_id,
                                   "article": article,
                                   "question": question,
                                   "options": options,
                                   "answer": answer,
                                   })
                current_batch_size += 1
                if current_batch_size == batch_size:
                    yield batch_data
                    current_batch_size = 0
                    batch_data = list()
    if current_batch_size > 0:
        yield batch_data


def yield_copa_sample(copa_path, filename, batch_size):
    current_batch_size = 0
    batch_data = list()
    data = ElementTree.parse(os.path.join(copa_path, filename))
    corpus = data.getroot()
    for item in corpus:
        id_ = item.attrib["id"]
        askfor = item.attrib["asks-for"]
        answer = int(item.attrib["most-plausible-alternative"])
        alternatives = list()
        premise = None
        for child in item:
            if child.tag == 'p':
                premise = child.text
            elif child.tag.startswith('a'):
                alternatives.append(child.text)
        assert premise is not None, id_
        assert len(alternatives) == 2, id_
        # @return id			: "1501"
        # @return premise		: "The item was packaged in bubble wrap."
        # @return answer		: 1
        # @return askfor		: "cause"
        # @return alternatives	: ["It was fragile.", "It arrived at its destination intact."]
        batch_data.append({"id": id_,
                           "premise": premise,
                           "askfor": askfor,
                           "answer": answer,
                           "alternatives": alternatives,
                           })
        current_batch_size += 1
        if current_batch_size == batch_size:
            yield batch_data
            current_batch_size = 0
            batch_data = list()
    if current_batch_size > 0:
        yield batch_data
