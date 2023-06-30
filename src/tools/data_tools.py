# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@163.sufe.edu.cn
import os
import json
import pandas


def yield_race_sample(race_path, types, difficulties, batch_size=1):
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
                assert len(questions) == len(options) == len(answers), f"{article_id}: mismatch of the number of questions({len(questions)}), options({len(options)}) and answers({len(answers)})!"
                # Ensure id matches filename
                assert article_id == difficulty + filename, f"{article_id}: Mismatch of id({article_id}) and filename({difficulty + filename})!"
                for question_id, (question, option, answer) in enumerate(zip(questions, options, answers)):
                    # Ensure total option is 4
                    n_option = len(option)
                    assert n_option == 4, f"{article_id}-{question_id}: {n_option} option!"
                    batch_data.append({"article_id": article_id,
                                       "question_id": question_id,
                                       "article": article,
                                       "question": question,
                                       "options": option,
                                       "answer": answer,
                                       })
                    current_batch_size += 1
                    # @return article_id	: "high1.txt"
                    # @return question_id	: 0
                    # @return article		: "My husband is a born shopper. He ..."
                    # @return question		: "The husband likes shopping because   _  ."
                    # @return options	    : ["he has much money.", "he likes the shops.", "he likes to compare the prices between the same items.", "he has nothing to do but shopping."]
                    # @return answer		: 'C'
                    if current_batch_size == batch_size:
                        yield batch_data
                        current_batch_size = 0
                        batch_data = list()
    if current_batch_size > 0:
        # Final batch
        yield batch_data
