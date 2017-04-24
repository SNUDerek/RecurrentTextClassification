import codecs, re
from nltk.corpus import brown

lower_lim = 8       # min size of input
upper_lim = 100     # max size of input
max_ex = 1000       # examples per genre
max_clusters = 3    # sentences per example, default = 1

# for sentence in brown.sents(categories=['religion'])[:100]:
#     print(' '.join(sentence))

sents = codecs.open('datasets/brown_sents.txt', 'w', encoding='utf-8')
classes = codecs.open('datasets/brown_topics.txt', 'w', encoding='utf-8')
'''
choose some from the following genres:
adventure
belles_lettres
editorial
fiction
government
hobbies
humor
learned
lore
mystery
news
religion
reviews
romance
science_fiction
'''
topics = ['adventure', 'religion', 'learned', 'government', 'humor', 'romance', 'lore',
          'mystery', 'news', 'science_fiction']
striplist = ["`", "'", '!', '?', '.', ',', ':', ';', '-', '(', ')', ]
counts_list = []

for topic in topics:
    good_count = 0 # for counting good sentences
    this_counter = 0
    this_cluster = ''
    for sentence in brown.sents(categories=[topic]):

        # check length first:
        if lower_lim < len(sentence) < upper_lim:

            this_string = ' '.join(sentence).lower() # lowercase
            for shit in striplist:
                this_string = this_string.replace(shit, '')     # remove punctuation etc
            this_string = re.sub(r'\d', '#', this_string) # sub # for digits
            this_string = re.sub(r'[\s]+', ' ', this_string)

            if this_counter < max_clusters:
                this_cluster += this_string
                this_counter += 1
            else:
                good_count += 1
                sents.write(this_cluster)
                sents.write('\n')
                classes.write(topic)
                classes.write('\n')
                this_counter = 0

            print(good_count, "sentence (clusters) for", topic)
            if good_count > max_ex:
                break

    counts_list.append(good_count)

print(sum(counts_list), counts_list)

