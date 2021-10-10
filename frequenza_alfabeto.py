import argparse
import logging
import time
import os
import string
import matplotlib.pyplot as plt

_description = 'Measure the releative frequencies of letters in a text file'

logging.basicConfig(filename='frequenza_alfabeto.log', level=logging.INFO)

def from_txt_to_str(txt):
    '''Apre e legge il file in ingresso'''
#logging.info('Opening input file %s', args.testo)
    file=open(txt, 'r')
    file.read()
    file.seek(0)
    file.close
    
#logging.info('Done. %d character(s) found', len(file))
    return file.read()

def print_dictonary(txt=None):
    ''' Riempie e stampa il dizionario con il numero di lettere'''
    dic={key:0 for key in string.ascii_lowercase}
    for val in txt:
        if val.isalpha():
            dic[val]=dic[val]+1
    i=i+1

def histogram(dict=None):
    '''Crea l'istogramma delle occorrenze'''
    if args.istogramma:
        plt.bar(list(dic.keys()), dic.values(), color='b')
        plt.show()
        
def time():
    '''Calcola il tempo necessario per svolgere l'intera operazione'''
    t0=time.time()
    dt=time.time()-t0
    print('Elapsed time: %.3f s' % dt)
print(dict)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='measure the relative frequency of the letters of the alphabet in a text')
    parser.add_argument('testo', help='insert the file .txt that you want to be analysed')
    parser.add_argument('-hist', '--histogram', help='print the histogram of the frequences", action="store_true')
    args=parser.parse_args()
    from_txt_to_str(args.testo)
    
    
    

