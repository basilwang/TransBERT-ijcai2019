import argparse
import os
import subprocess
import sys


LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "..", "lib")


def lemmatize_tagged(word_tag_pairs):
    input_string = " ".join(["%s_%s" % (word, tag) for (word, tag) in word_tag_pairs])

    if input_string:
        # Use the bash wrapper to do lemmatization
        proc = subprocess.Popen([os.path.join(LIB_DIR, "morph/morpha")],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_text, err_text = proc.communicate(input_string)
        if not out_text:
            raise LemmatizerError("lemmatizer failed: %s" % err_text)
        else:
            return out_text.strip("\n")
    else:
        return ""


def lemmatize_words(words):
    input_string = " ".join(words)

    if input_string:
        # Use the bash wrapper to do POS tagging and lemmatization
        proc = subprocess.Popen([os.path.join(LIB_DIR, "morpha.sh")],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_text, err_text = proc.communicate(input_string)
        if not out_text:
            raise LemmatizerError("lemmatizer failed: %s" % err_text)
        else:
            return out_text.strip("\n")
    else:
        return ""


def lemmatize_verb(verb):
    if verb:
        # Use the bash wrapper to do POS tagging and lemmatization
        proc = subprocess.Popen([os.path.join(LIB_DIR, "morpha.sh")],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_text, err_text = proc.communicate("%s_V" % verb)
        if not out_text:
            raise LemmatizerError("lemmatizer failed: %s" % err_text)
        else:
            return out_text.strip("\n")
    else:
        return ""


def morpha(words, poses):
    if words:
        # Prepare input string
        # Need to add a full stop for morpha to work right
        input_string = "%s ." % " ".join("%s_%s" % (word, pos) for (word, pos) in zip(words, poses))

        # Call the morpha binary
        proc = subprocess.Popen([os.path.join(LIB_DIR, "morph/morpha"), "-a"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_text, err_text = proc.communicate(input_string)

        if not out_text:
            raise LemmatizerError("lemmatizer failed: %s" % err_text)
        else:
            out_text = out_text.strip("\n")
            # Leave off the full stop we added
            words = out_text.split(" ")[:-1]
            # Split up the affixes
            return [word_out.split("+") for word_out in words]
    else:
        return []


def is_plural(noun):
    morph_anal = morpha([noun], ["N"])[0]
    # Check whether there's a "s" in the affixes
    return "s" in morph_anal[1:]


def inflect_verb(verb, plural=False, past=False, passive=False):
    prefix = ""
    if verb == "be":
        # Handling "be" with morphg involves so many special cases that there's no point in calling it at all
        if past:
            if plural:
                inflected_verb = "were"
            else:
                inflected_verb = "was"
        else:
            if plural:
                inflected_verb = "are"
            else:
                inflected_verb = "is"
    else:
        # Prepare input string for morphg
        if passive:
            if past and plural:
                prefix = "were "
            elif past:
                prefix = "was "
            elif plural:
                prefix = "are "
            else:
                prefix = "is "
            affix = "+en"
        elif past:
            affix = "+ed"
        elif not plural:
            affix = "+s"
        else:
            affix = ""
        # Need to add a full stop for morpha to work right
        input_string = "%s%s_V ." % (verb, affix)

        # Call the morphg binary
        proc = subprocess.Popen([os.path.join(LIB_DIR, "morphg")],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_text, err_text = proc.communicate(input_string)

        if not out_text:
            raise MorphgError("morphg failed: %s" % err_text)
        else:
            inflected_verb = out_text.strip("\n").split()[0]

    return "%s%s" % (prefix, inflected_verb)


class LemmatizerError(Exception):
    pass


class MorphgError(Exception):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test lemmatization functions. Words should be separated from their POS tags by |s (C&C style)")
    subparsers = parser.add_subparsers(help="Tools", dest="tool")

    lemmatize_parser = subparsers.add_parser("lemmatize", help="Lemmatize words")
    lemmatize_parser.add_argument("--pos", action="store_true",
                                  help="POS tag prior to lemmatization (POS tags must be given otherwise)")

    anal_parser = subparsers.add_parser("anal", help="Morphological analysis")
    anal_parser.add_argument("--nouns", action="store_true",
                             help="Assume all input words are nouns -- otherwise POS tags must be given")

    opts = parser.parse_args()

    if opts.tool == "lemmatize":
        print >>sys.stderr, "Accepting input from stdin"
        for lines in sys.stdin.readlines():
            for line in lines.splitlines():
                if opts.pos:
                    print lemmatize_words(line.split())
                else:
                    word_tag_pairs = [word_tag.split("|") for word_tag in line.split()]
                    if not all(len(pair) == 2 for pair in word_tag_pairs):
                        print >>sys.stderr, "Could not split input into words and tags: they should be separated by '|'"
                        sys.exit(1)
                    print lemmatize_tagged(word_tag_pairs)
    elif opts.tool == "anal":
        print >>sys.stderr, "Accepting input from stdin"
        for line in sys.stdin:
            words = line.split()
            if opts.nouns:
                poses = ["N"] * len(words)
            else:
                words, __, poses = zip(*(s.partition("_") for s in words))
            print morpha(words, poses)