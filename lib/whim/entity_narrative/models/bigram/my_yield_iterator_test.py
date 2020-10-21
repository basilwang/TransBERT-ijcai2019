class RichEventDocumentCorpus(object):
    def __iter__(self):
        for filename in range(10):
            yield filename


richEventDocumentCorpus = RichEventDocumentCorpus()
for r in enumerate(richEventDocumentCorpus):
    print r

for r in richEventDocumentCorpus:
    print r


enumerate(richEventDocumentCorpus).next()
