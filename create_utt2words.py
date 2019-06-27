import sys
from collections import defaultdict

import plac
from loguru import logger


class UttInfo:
    def __init__(self, uttid, startidx, endidx):
        self.uttid = self.uttid
        self.startidx = self.startidx
        self.endidx = self.endidx


def remove_repeats(lst):
    newlst = [lst[0]]
    for i in range(1, len(lst)):
        val = lst[i]
        if val != newlst[-1]:
            newlst.append(val)
    return newlst


def get_repition_counts(lst):
    newlst = [1]
    curval = lst[0]
    idx = 0
    for i in range(1, len(lst)):
        if curval == lst[i]:
            newlst[idx] += 1
        else:
            curval = lst[i]
            newlst.append(1)
            idx += 1
    return newlst


def main(fpath_phoneali, fpath_text, fpath_lexicon, fpath_phonestable, fpath_featsidx, outf, debug: ('debug mode', 'flag', 'd')):
    logger.remove()
    if debug:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")
    phone_map = {}
    with open(fpath_phonestable) as fh:
        for line in fh:
            line = line.split()
            num = int(line[1])
            phone_map[line[0]] = num
            phone_map[num] = line[0]

    utt2phones = {}
    with open(fpath_phoneali) as fh:
        for line in fh:
            uttid, *phones = line.split()
            phone_lengths = get_repition_counts(phones)
            # This only works because of using pos dependent phones! Otherwise two words with the same end & start phone
            # would have their respective phones merged together into one.
            phones_norepeats = remove_repeats(phones)
            utt2phones[uttid] = ([phone_map[int(e)] for e in phones_norepeats], phone_lengths)

    utt2words = defaultdict(list)
    with open(fpath_text) as fh:
        for line in fh:
            uttid, *words = line.split()
            utt2words[uttid] = words

    utt2indcs = defaultdict(list)
    with open(fpath_featsidx) as fh:  # should be ordered
        lastidx = -1
        for line in fh:
            uttid, idx, startidx, endidx = line.split()
            idx = int(idx)
            assert idx > lastidx
            utt2indcs[uttid].append((int(startidx), int(endidx),))
            lastidx = idx

    word2pron = defaultdict(list)
    with open(fpath_lexicon) as fh:
        for line in fh:
            word, *pron = line.split()
            if len(pron) != 1:
                pron[0] += '_B'
                for i in range(1, len(pron) - 1):
                    pron[i] += '_I'
                pron[-1] += '_E'
            else:
                pron[0] += '_S'
            word2pron[word].append(remove_repeats(pron))

    with open(outf, 'w') as fhw:
        for uttid, words in utt2words.items():
            fhw.write(uttid + ' 1')
            if uttid not in utt2phones:
                continue
            phones, phone_lengths = utt2phones[uttid]
            indcs = utt2indcs[uttid]
            if len(indcs) == 0:
                continue
            chunk_idx = 0
            frame_idx = 0

            lastphone = '#NONE'
            lastword = '#NONE'

            for word in words:
                startidx, endidx = indcs[chunk_idx]
                logger.debug('%s %s %d %d %d' % (uttid, word, frame_idx, startidx, endidx))
                prons = word2pron[word]
                pron = prons[0]
                numphones = len(pron)
                if len(phones) == 0:
                    fhw.write(f' {word}')
                    break
                logger.debug('B %s % s %s %s' %(lastword, lastphone, phones[0], pron))
                if lastphone == pron[0] and len(pron) == 1 and pron[0] != phones[0] and not ('sil' in phones[0] and phones[1] == pron[0]):
                    # to deal with situations like two 'I's in sequence
                    pass
                else:
                    while phones[:numphones] != pron:
                        if 'sil' not in phones[0] and 'jnk' not in phones[0]:
                            if len(prons) > 1:
                                pron = prons[1]
                                numphones = len(pron)
                                if pron[0] == phones[0]:
                                    break
                        assert ('sil' in phones[0] or 'jnk' in phones[0]), '{} {}'.format(phones[:10], pron)
                        step = phone_lengths[0]
                        phones = phones[1:]
                        phone_lengths = phone_lengths[1:]
                        frame_idx += step
                    logger.debug(pron[:numphones])
                    assert pron == phones[:numphones], '{} {}'.format(pron, phones[:numphones])
                    step = sum(phone_lengths[:numphones])
                    phone_lengths = phone_lengths[numphones:]
                    lastphone = phones[numphones-1]
                    phones = phones[numphones:]
                    frame_idx += step
                if frame_idx > endidx + 1:
                    if chunk_idx < len(indcs) - 1:
                        chunk_idx += 1
                    fhw.write(f'\n{uttid} {chunk_idx+1} {word}')
                else:
                    fhw.write(f' {word}')
                lastword = word
            fhw.write('\n')


plac.call(main)