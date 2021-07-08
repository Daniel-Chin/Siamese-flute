##### cell 0 #####
# FILE_MUS = 'test0.mp3'
# FILE_CSV = 'test0.csv'

# # adjust until good alignment
# OFFSET = 848
# STRETCH = -.00003
# LEFT_TEST = 420, 620
# RIGHT_TEST = 3850, 3950

# TRIM_START = 400
##### cell 1 #####
FILE_MUS = 'test1.mp3'
FILE_CSV = 'test1.csv'

# adjust until good alignment
OFFSET = -490
STRETCH = -.00004
LEFT_TEST = 50, 190
RIGHT_TEST = 9300, 9600

TRIM_START = 30
##### cell 2 #####
import numpy as np
from numpy import pi
from matplotlib import pyplot as plt
import librosa
from IPython.display import Audio
import scipy
import csv
##### cell 3 #####
PAGE_LEN = 2048

HOP_LEN = PAGE_LEN // 4
##### cell 4 #####
def sino(freq, length):
    return np.sin(np.arange(length) * freq * TWO_PI / SR)

def play(data):
    return Audio(np.concatenate([data, [1]]), rate = SR)

def findPeaks(energy):
    slope = np.sign(energy[1:] - energy[:-1])
    extrema = slope[1:] - slope[:-1]
    return np.argpartition(
        (extrema == -2) * energy[1:-1], - N_HARMONICS,
    )[- N_HARMONICS:] + 1

def sft(signal, freq_bin):
    # Slow Fourier Transform
    return np.abs(np.sum(signal * np.exp(IMAGINARY_LADDER * freq_bin))) / PAGE_LEN

def refineGuess(guess, signal):
    def loss(x):
        if x < 0:
            return 0
        return - sft(signal, x)
    freq_bin, loss = blindDescend(loss, .01, .4, guess)
    return freq_bin * SR / PAGE_LEN, - loss

def widePlot(h = 3, w = 12):
    plt.gcf().set_size_inches(w, h)

    
def spectro(signal, do_wide = True, trim = 130):
    energy = np.abs(rfft(signal * HANN))
    plt.plot(energy[:trim])
    if do_wide:
        widePlot()

def concatSynth(synth, harmonics, n):
    buffer = []
    for i in range(n):
        synth.eat(harmonics)
        buffer.append(synth.mix())
    return np.concatenate(buffer)

def pitch2freq(pitch):
    return np.exp((pitch + 36.37631656229591) * 0.0577622650466621)

def freq2pitch(f):
    return np.log(f) * 17.312340490667562 - 36.37631656229591
##### cell 5 #####
raw_0, SR = librosa.load(FILE_MUS)
SR
##### cell 6 #####
# play(raw_0)
##### cell 7 #####
# help(librosa.yin)
f0s = librosa.yin(raw_0, 200, 2500, SR, PAGE_LEN)
#plt.plot(f0s)
#widePlot()
##### cell 8 #####
def traceEnergy(signal):
    i = 0
    energy = []
    while True:
        page = signal[i*HOP_LEN : i*HOP_LEN + PAGE_LEN]
        if page.size < PAGE_LEN:
            break
        energy.append(np.sum(scipy.signal.periodogram(page, SR)) / PAGE_LEN)
        i += 1
    return energy
e = np.array(traceEnergy(raw_0))
#plt.plot(e)
#widePlot()
##### cell 9 #####
ee = (e - 2758.94165039096) * 10000000
#plt.plot(ee)
#widePlot()
##### cell 10 #####
def getP():
    time = []
    pressure = []
    with open(FILE_CSV, 'r') as f:
        last_t = -1
        epoch = 0
        for t, p in csv.reader(f):
            t = int(t)
            if t < last_t:
                epoch += 1
            last_t = t
            time.append((t + 16384 * epoch) / 1000)
            pressure.append(int(p))
    return time, pressure
t, p = getP()
#plt.plot(t, p)
#widePlot()
##### cell 11 #####
def sampleP(time, pressure, t):
    s = np.sign(time - t)
    i = np.where(s[1:] - s[:-1])[0][0]
    t4, t5, t6 = time[i], t, time[i+1]
    return pressure[i] * (t6-t5) / (t6-t4) + pressure[i+1] * (t5-t4) / (t6-t4)

def uniformP(time, pressure):
    time = np.array(time)
    t = 0
    result = []
    while True:
#         print(t, end='\r', flush = True)
        t += HOP_LEN / SR + STRETCH
        if t > time[-1]:
            break
        if t < time[0]:
            continue
        result.append(sampleP(time, pressure, t))
#     print('Done                ')
    return np.array(result)
pp = uniformP(t, p)
##### cell 12 #####
if OFFSET > 0:
    eee = ee[OFFSET:]
    ff = f0s[OFFSET:]
    pp_ = pp
else:
    pp_ = pp[-OFFSET:]
    eee = ee
    ff = f0s
##### cell 13 #####
st, en = LEFT_TEST
#plt.plot(pp_[st:en])
#plt.plot(eee[st:en] * 3)
#widePlot()
##### cell 14 #####
st, en = RIGHT_TEST
#plt.plot(pp_[st:en])
#plt.plot(eee[st:en] * 3)
#widePlot()
##### cell 15 #####
#plt.plot(eee[:1500])
#widePlot()
##### cell 16 #####
eee = eee[:pp_.size]
ff = ff[:pp_.size]
eeee = eee[TRIM_START:]
fff = ff[TRIM_START:]
ppp = pp_[TRIM_START:]
##### cell 17 #####
ffff = []
for x, y in zip(fff, ppp):
    if y > 15:
        ffff.append(x)
    else:
        ffff.append(0)
ffff = np.array(ffff)
#plt.plot(ffff)
#widePlot()
##### cell 18 #####
#plt.plot(eeee * 18)
#plt.plot(ppp * 8)
#plt.plot(ffff)
#widePlot(10, 150)
#plt.savefig('eyeball.pdf')
eeee.size, ppp.size, ffff.size
##### cell 19 #####
eeee[eeee<0] = 0
#plt.plot(eeee ** .33 * 40)
#plt.plot(ppp)
#plt.plot(ffff * .1)
#widePlot(10, 150)
#plt.savefig('eyeball_1_3.pdf')
eeee.size, ppp.size, ffff.size
##### cell 20 #####
def scatterBend(p, f, start, end, pitch, c):
    p = p[start:end]
    f = f[start:end]
    pb = freq2pitch(f) - pitch
    pp = []
    pbpb = []
    for x, y in zip(p, pb):
        if abs(y) < 3:
            pp.append(x)
            pbpb.append(y)
    plt.scatter(pp, pbpb, c=c, s=.5)

##### cell 21 #####
plt.plot(ffff)
widePlot()
plt.show()
exit()
##### cell 22 #####
SCATTER = [
    [580, 1300, 83, 'r'],
    [1550, 2200, 82, 'g'],
    [2200, 2800, 80, 'b'],
    [2800, 3450, 78, 'k'],
]
##### cell 23 #####
for args in SCATTER:
    scatterBend(ppp, ffff, *args)
plt.grid(which='major')
##### cell 24 #####
# Legacy code

# plt.plot(ffff[230:1300])
# # plt.plot(ffff[1550:2200])
# # plt.plot(ffff[2200:2800])
# # plt.plot(ffff[2800:3450])
# widePlot()

# plt.plot(ffff[230:580])
# plt.plot(ffff[580:960])

# # scatterBend(ppp, ffff, 230, 580, 83, 'r')

# scatterBend(ppp, ffff, 580, 1300, 83, 'r')
# scatterBend(ppp, ffff, 1550, 2200, 82, 'g')
# scatterBend(ppp, ffff, 2200, 2800, 80, 'b')
# scatterBend(ppp, ffff, 2800, 3450, 78, 'k')
# plt.grid(which='major')
