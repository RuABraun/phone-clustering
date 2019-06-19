import plac
import numpy as np
import subprocess as sp
import os
import python_rw


def calc_feats(datadir, aug_datadir):
    if not os.path.exists(f'{datadir}/feats.scp'):
        cmd = f'cd /work; steps/make_fbank.sh --nj 8 --cmd util/queue.pl --fbank-config conf/fbank.conf {datadir}'
        sp.check_output(cmd, shell=True)

    if not os.path.exists(f'{aug_datadir}/feats.scp'):
        cmd = f'cd /work; LC_ALL=en_GB.UTF-8 python3 steps/data/augment_additive.py --include-prob 1.0 --utt-suffix "" --fg-snrs "13:11:9:7" --fg-noise-dir /data/noise/musan/noise --mg-snrs "9:11:13" --mg-noise-dir /data/noise/musan/music --bg-snrs "21:19:17:15" --num-bg-noises "3:4:5:6" --bg-noise-dir /data/noise/musan/speech {datadir} {aug_datadir}'
        sp.check_output(cmd, shell=True)
        cmd = f'cd /work; steps/make_fbank.sh --nj 8 --cmd utils/queue.pl --fbank-config conf/fbank.conf {aug_datadir}'
        sp.check_output(cmd, shell=True)


def select_feats(arkf, phonealif, keys, outf):
    data = []
    feats_reader = python_rw.Pykread(arkf)
    cnt = 0
    with open(phonealif) as fh_ali:
        ismore = True
        while ismore:
            if feats_reader.done():
                break
            key, mat = feats_reader.get()
            if key not in keys:
                ismore = feats_reader.next()
                continue
            cnt += 1
            lineali = next(fh_ali)
            lineali = lineali.split()
            assert key == lineali[0], '{} {}'.format(key, lineali[0])
            lineali = [int(v) for v in lineali[1:]]
            data_arr = []
            num = min(len(lineali), mat.shape[0])
            if (len(lineali) - mat.shape[0]) ** 2 > 1:
                print('{} {}'.format(len(lineali), mat.shape[0]))
                raise RuntimeError('Lengths do not match!')

            for i, t in enumerate(lineali):
                if i >= num:
                    break
                if t > 10:  # not silence or jnk
                    data_arr.append(mat[i])
                else:
                    if len(data_arr) > 4:
                        data.append(np.array(data_arr))
                    data_arr = []
            if len(data_arr) > 4:
                data.append(np.array(data_arr))
            ismore = feats_reader.next()
    print(f'Found {cnt}')
    feats_reader.close()
    dataf = outf + '.npy'
    np.save(dataf, data)
    num_chunks = len(data)
    print(f'Saved {dataf}, found {num_chunks}')


def main(datadir, alif, workdir, outf_base):
    """ Extracts contiguous non silence features . """
    model = os.path.dirname(alif) + '/final.mdl'

    aug_datadir = '{}_augmented'.format(datadir.rstrip('/'))
    calc_feats(datadir, aug_datadir)

    # convert ali to phones
    cmd = f'ali-to-phones --per-frame=true {model} "ark:gunzip -c {alif}|" ark,scp:{workdir}/phoneali,{workdir}/phoneali.scp'
    sp.check_output(cmd, shell=True)
    sp.check_output(f"sort -u {workdir}/phoneali.scp -o {workdir}/phoneali.scp", shell=True)
    cmd = f'copy-int-vector scp:{workdir}/phoneali.scp ark,t:{workdir}/phoneali_sorted.txt'
    sp.check_output(cmd, shell=True)

    keys = set()
    with open(f'{workdir}/phoneali.scp') as fh:
        for line in fh:
            keys.add(line.split()[0])
    print(f'Num keys ' + str(len(keys)))

    cmd = f'sort -u {datadir}/feats.scp -o {datadir}/feats_sorted.scp'
    sp.check_output(cmd, shell=True)
    cmd = f'copy-feats scp:{datadir}/feats_sorted.scp ark:{datadir}/data/raw_fbank_all.ark'
    sp.check_output(cmd, shell=True)
    select_feats(f'ark:{datadir}/data/raw_fbank_all.ark', f'{workdir}/phoneali_sorted.txt', keys,
                 outf_base)

    cmd = f'sort -u {aug_datadir}/feats.scp -o {aug_datadir}/feats_sorted.scp'
    sp.check_output(cmd, shell=True)
    cmd = f'copy-feats scp:{aug_datadir}/feats_sorted.scp ark:{aug_datadir}/data/raw_fbank_all.ark'
    sp.check_output(cmd, shell=True)
    select_feats(f'ark:{aug_datadir}/data/raw_fbank_all.ark', f'{workdir}/phoneali_sorted.txt', keys,
                 outf_base + '_aug')


plac.call(main)