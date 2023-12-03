import pandas as pd
import re

# Collect the sentences starting with A man or A woman
sentences = []
with open('STS-B/sts-test.csv', 'rb') as f:
    text = f.readline()
    while text:
        text = str(text).replace('\\n','').replace("'",'').replace('"','').strip().split('\\t')
        for s in text[5:7]:
            if 'a man' in s.lower().strip()[:5]:
                sentences.append(s[6:])
            elif 'a woman' in s.lower().strip()[:7]:
                sentences.append(s[8:])
                
        text = f.readline()

print(f'Total pair of sentences: {len(sentences)}')

# Retrieve the list of gendered words
gendered_words = []
for g in ['male', 'female']:
    with open(f'STS-B/{g}_word_file.txt', 'rb') as f:
        words = f.readlines()
        words = [str(w).replace('\\n','').replace("'",'').replace('"','').replace('b','').replace(' ','') for w in words]
        gendered_words.extend(words)

# Use the list to remove them from the initial sentences
sentences_final = []
for s in sentences:
    if not any([(w.lower() in s.lower().split()) or (w.lower()+'s' in s.lower().split()) for w in gendered_words]):
        sentences_final.append(s)

# Remove duplicates
sentences_final = list(set(sentences_final))
print(f'Total pair of sentences after removing gendered words: {len(sentences_final)}')

# Obtain occupations from https://github.com/rudinger/winogender-schemas/blob/master/data/occupations-stats.tsv
occupations = []
with open(f'STS-B/occupations-stats.tsv', 'rb') as f:
    words = f.readlines()
    words = [str(w).replace('\\n','').replace("'",'').replace('"','').replace('b','').replace(' ','').split('\\')[0] for w in words]
    occupations.extend(words)
occupations.remove('occupation')

# Create the final pairs of sentences
final_set = []

for t in sentences_final:
    for o in occupations:
        new_s = f'A {o} ' + t
        
        # Replace the article
        if re.match(r'[aeiouh]', o, re.IGNORECASE):
            new_s = new_s.replace('A ', 'An ')

        final_set.append(['A man '+t, new_s])
        final_set.append(['A woman '+t, new_s])

# Save the final set
df = pd.DataFrame(final_set, columns=['sent1', 'sent2'])
df.to_csv('sts-test-bias-final.csv', index=False, sep='\t')