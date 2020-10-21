from numpy import zeros, dtype, float32 as REAL, fromstring

from gensim import utils  # utility fnc for pickling, common scipy operations etc
from gensim.models.word2vec import Word2Vec, Vocab


def read_word(f):
    # mixed text and binary: read text first to get word
    word = []
    while True:
        ch = f.read(1)
        if ch == '':
            # EOF: give up reading
            return None
        elif ch == b' ':
            break
        elif ch != b'\n':  # ignore newlines in front of words (some binary files have newline, some don't)
            word.append(ch)
    word = utils.to_unicode(b''.join(word))
    return word


def load_word2vec_format_filtered(fname, vocab, fvocab=None, binary=False, norm_only=True):
    """
    Like Word2Vec's loader, but allows you to restrict to a limited vocabulary.

    """
    vocab = set(vocab)
    counts = None
    if fvocab is not None:
        counts = {}
        with utils.smart_open(fvocab) as fin:
            for line in fin:
                word, count = utils.to_unicode(line).strip().split()
                counts[word] = int(count)

    with utils.smart_open(fname) as fin:
        header = utils.to_unicode(fin.readline())
        vocab_size, layer1_size = map(int, header.split())  # throws for invalid file format
        # We know we only need to store the number of things in the vocab
        vocab_size = len(vocab)

        result = Word2Vec(size=layer1_size)
        result.syn0 = zeros((vocab_size, layer1_size), dtype=REAL)
        word_num = 0
        if binary:
            binary_len = dtype(REAL).itemsize * layer1_size
            while word_num < vocab_size:
                # mixed text and binary: read text first, then binary
                word = read_word(fin)
                if word is None:
                    # Reached EOF
                    break
                # Only store the vectors for words in the given vocabulary
                if word in vocab:
                    vocab.remove(word)
                    if counts is None:
                        result.vocab[word] = Vocab(index=word_num, count=vocab_size - word_num)
                    elif word in counts:
                        result.vocab[word] = Vocab(index=word_num, count=counts[word])
                    else:
                        result.vocab[word] = Vocab(index=word_num, count=None)
                    result.index2word.append(word)
                    result.syn0[word_num] = fromstring(fin.read(binary_len), dtype=REAL)
                    word_num += 1
                else:
                    # Skip this vector
                    fin.read(binary_len)
        else:
            for line_no, line in enumerate(fin):
                parts = utils.to_unicode(line).split()
                if len(parts) != layer1_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, weights = parts[0], map(REAL, parts[1:])
                if word in vocab:
                    vocab.remove(word)
                    if counts is None:
                        result.vocab[word] = Vocab(index=word_num, count=vocab_size - word_num)
                    elif word in counts:
                        result.vocab[word] = Vocab(index=word_num, count=counts[word])
                    else:
                        result.vocab[word] = Vocab(index=word_num, count=None)
                    result.index2word.append(word)
                    result.syn0[word_num] = weights
                    word_num += 1
                    if word_num >= vocab_size:
                        # Got all we need: don't carry on reading
                        break
    # Get rid of the empty vectors at the end if not all words were found
    if word_num < vocab_size:
        result.syn0 = result.syn0[:word_num].copy()
    result.init_sims(norm_only)
    return result