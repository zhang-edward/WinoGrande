import spacy
import en_core_web_lg

parser = en_core_web_lg.load()

doc = parser("Ian volunteered to eat Dennis's menudo after already having a bowl because _ despised eating intestine")
print(doc.to_json())
