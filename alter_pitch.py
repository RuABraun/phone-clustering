import plac
import random

def main(f):
    newlines = []
    with open(f) as fh:
        for line in fh:
            wavid, wav = line.split()
            pitch = None
            if random.randint(0, 1) == 0:
                pitch = random.randint(150, 250)
            else:
                pitch = random.randint(-250,-150)
            pitch = str(pitch)
            line = f'{wavid} sox {wav} -r 16000 -c 1 -t wavpcm - pitch {pitch} |'
            newlines.append(line)

    with open(f, 'w') as fh:
        for l in newlines:
            fh.write(f'{l}\n')


plac.call(main)