import spacy

nlp = spacy.load("en_core_web_sm")

def extract_order_info(sentence):
    doc = nlp(sentence)
    order_info = {}

    for token in doc:
        if token.text.lower() == "đặt":
            for child in token.children:
                if child.dep_ == "dobj":
                    order_info["name"] = child.text

        if token.pos_ == "NUM" and token.dep_ == "nummod":
            order_info["number"] = token.text

    return order_info

sentence = "Đặt tôi 3 đùi gà KFC"
order_info = extract_order_info(sentence)
print(order_info)
