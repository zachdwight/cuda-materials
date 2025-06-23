#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  queryGenome_with_CUDA.py
#  
#  Copyright 2019 Zachary Dwight
#  
#  
# 



import warnings
warnings.filterwarnings('ignore')
import numba
from numba import cuda
import numpy as np
from numba import vectorize
import os

os.environ['NUMBAPRO_NVVM'] = r'/home/user/anaconda2/lib/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = r'/user/zachary/anaconda2/lib/'

productsize = 300
complement = {'A':'T','C':'G','G':'C','T':'A'}

def revcomp(sequence):
	bases = list(sequence)
	bases = reversed([complement.get(base,base) for base in bases])
	bases = ''.join(bases)
	return bases
	
def mismatches(sequence):
	mm = []
	s=''
	mm.append(sequence)
	for i in range(len(sequence)):
		if(i<13):
			if (sequence[i]=='A'):
				s = sequence[:i]+'T'+sequence[i+1:]
				mm.append(s)
				s = sequence[:i]+'G'+sequence[i+1:]
				mm.append(s)
				s = sequence[:i]+'C'+sequence[i+1:]
				mm.append(s)
			if (sequence[i]=='T'):
				s = sequence[:i]+'A'+sequence[i+1:]
				mm.append(s)
				s = sequence[:i]+'G'+sequence[i+1:]
				mm.append(s)
				s = sequence[:i]+'C'+sequence[i+1:]
				mm.append(s)
			if (sequence[i]=='G'):
				s = sequence[:i]+'A'+sequence[i+1:]
				mm.append(s)
				s = sequence[:i]+'T'+sequence[i+1:]
				mm.append(s)
				s = sequence[:i]+'C'+sequence[i+1:]
				mm.append(s)
			if (sequence[i]=='C'):
				s = sequence[:i]+'A'+sequence[i+1:]
				mm.append(s)
				s = sequence[:i]+'T'+sequence[i+1:]
				mm.append(s)
				s = sequence[:i]+'G'+sequence[i+1:]
				mm.append(s)
	return mm
	
			
def bulges(sequence):
	bgs = []
	s = sequence[-14:]
	bgs.append(sequence)
	for i in range(len(s)):
		if(i<13 and i>0):
			s = sequence[:i]+''+sequence[i+1:]
			bgs.append('A'+s)
			bgs.append('T'+s)
			bgs.append('G'+s)
			bgs.append('C'+s)
	for i in range(len(s)):
		if(i<13 and i>0):
			Ai = sequence[:i]+'A'+sequence[i:]
			Ti = sequence[:i]+'T'+sequence[i:]
			Gi = sequence[:i]+'G'+sequence[i:]
			Ci = sequence[:i]+'C'+sequence[i:]
			bgs.append(Ai[-14:])
			bgs.append(Ti[-14:])
			bgs.append(Gi[-14:])
			bgs.append(Ci[-14:])
	return bgs

@numba.jit
def sub_gpu(a,b):
	d = b - a
	if d<300 and d>0:
		return d
vfunc = np.vectorize(sub_gpu,otypes=[np.float64])

def cuda_sub(a,b):
	d = b - a
	return d
cfunc = np.vectorize(cuda_sub,otypes=[np.float64])

def cuda_query(fmm,rmm):
	chrs = ['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrx','chry','chrm']

	conn = sqlite3.connect('UTAH_GENOME')
	c = conn.cursor()
	p = ''
	for fm in fmm:
		for rm in rmm:
			fquery = ''
			rquery = ''

			
			for chrm in chrs:
				trimc = chrm.replace('chr','')
				fquery += "SELECT *, '{}' as 'chr' FROM {} WHERE seq='{}' ".format(trimc,chrm,fm)
				rquery += "SELECT *, '{}' as 'chr' FROM {} WHERE seq='{}' ".format(trimc,chrm,rm)
				if chrm != chrs[-1]:
					fquery += "UNION ALL "
					rquery += "UNION ALL "
			
			c.execute(fquery)
			fresult = c.fetchall()
			
			if (fresult):
				c.execute(rquery)
				rresult = c.fetchall()
				if(rresult):
					for frow in fresult:
						
						for rrow in rresult:
							chrA = str(frow[2])
							chrB = str(rrow[2])
							if (chrA == chrB):
								flocs = frow[1]
								rlocs = rrow[1]
								findexes = [int(x) for x in flocs.split(',')]
								rindexes = [int(x) for x in rlocs.split(',')]
								for f in findexes:
									for r in rindexes:
										dif = r - f
										if(dif<=productsize and dif>0):
											st = f - len(fp) + 14
											end = r + len(rp)
											sizebp = abs(st - end)
											r = r + 14
											f = r - sizebp + 1
											p += '     >>>> chr{} {}bp Product Found (F+R) >> {} to {} with {}bp spacing'.format(chrA,int(sizebp),int(f),int(r),int(dif))
											p += '     ------------ F: {} & R: {}'.format(fm,rm)
	return p
qfunc = np.vectorize(cuda_query,otypes=[np.string_])


import time
start = time.time()

allchrs = ['chr1','chr2','chr3','chr4','chr5','chr6','chr7','chr8','chr9','chr10','chr11','chr12','chr13','chr14','chr15','chr16','chr17','chr18','chr19','chr20','chr21','chr22','chrx','chry','chrm']
chra = ['chr1','chr6','chr11','chr16','chr21']
chrb = ['chr2','chr7','chr12','chr17','chr22']
chrc = ['chr3','chr8','chr13','chr18','chrx']
chrd = ['chr4','chr9','chr14','chr19','chry']
chre = ['chr5','chr10','chr15','chr20','chrm']

#10518 - chr4
fp = 'AAATATATCAATGTAAGAAGTTCACTCTTAGAC'
rp = 'CCCATGTGACCTCAGAACAC'

#9419
fp = 'AGGGCTACTCCGGAAGC'
rp = 'GGCTCCTCTGCACAAGAG'

#LOTS OF HITS TEST
#fp = 'GAAAAGAACAAAAG'
#rp = 'GTTCTTTTGTTTTC'


fplen = len(fp) - 14
rplen = len(rp) - 14
former = fp[-14:]
revmer = revcomp(rp[-14:]) 
print 'Genome Product Query v2'
print '__________________'
print "Forward 5-3' : {}".format(fp)
print "Reverse 5-3' : {}".format(rp)
print '__________________'


import sqlite3

def PerfectSearch(chrs):
	####  PERFECT PRODUCTS
	conn = sqlite3.connect('UTAH_GENOME')
	c = conn.cursor()
	print '|| PERFECT PRODUCT'
	for chrm in chrs:
		rlocs = ''
		flocs = ''
		
		query = "SELECT * FROM {} WHERE seq='{}'".format(chrm,former)
		c.execute(query)
		fresult = c.fetchone()
		if(fresult):
			flocs = fresult[1]
		
		query = "SELECT * FROM {} WHERE seq='{}'".format(chrm,revmer)
		c.execute(query)
		rresult = c.fetchone()
		if(rresult):
			rlocs = rresult[1]
			
		findexes = flocs.split(',')
		rindexes = rlocs.split(',')
		
		if(len(flocs)>0 and len(rlocs)>0):
			for f in findexes:
				for r in rindexes:
					dif = float(r) - float(f)
					if(dif<=productsize and dif>0):
					
						dif = dif - 14
						st = float(f) - len(fp) + 14
						end = float(r) + len(rp)
						sizebp = abs(st - end)
						r = float(r) + 14
						f = r - sizebp + 1
						print '     >>>> {} {}bp Product Found (F+R) >> {} to {} with {}bp spacing'.format(chrm,int(sizebp),int(f),int(r),int(dif))
		
		if(len(flocs)>0):
			for f in findexes:
				for r in findexes:
					dif = float(r) - float(f)
					if(dif<=productsize and dif>0):
						dif = dif - 14
						st = float(f) - len(fp) + 14
						end = float(r) + len(rp)
						sizebp = abs(st - end)
						r = float(r) + 14
						f = r - sizebp + 1
						print '     >>>> {} {}bp Product Found (F+F) >> {} to {} with {}bp spacing'.format(chrm,int(sizebp),int(f),int(r),int(dif))
		
						
		if(len(rlocs)>0):
			for f in rindexes:
				for r in rindexes:
					dif = float(r) - float(f)
					if(dif<=productsize and dif>0):
						dif = dif - 14
						st = float(f) - len(fp) + 14
						end = float(r) + len(rp)
						sizebp = abs(st - end)
						r = float(r) + 14
						f = r - sizebp + 1
						print '     >>>> {} {}bp Product Found (R+R) >> {} to {} with {}bp spacing'.format(chrm,int(sizebp),int(f),int(r),int(dif))
	end = time.time()
	seconds = end - start
	print 'Perfect PCR search executed in {} seconds.'.format("%.3f" % seconds)



def MOBSearchCUDA(search):
	begtime = time.time()

	outmsg = ''
	if search=='m':
		fmm = mismatches(former)
		rmm = mismatches(revmer)
		outmsg = 'MxM'
	if search=='b':
		fmm = bulges(former)
		rmm = bulges(revmer)
		outmsg = 'BxB'
	if search=='d':
		fmm = bulges(former)
		fmm.extend(mismatches(former))
		rmm = bulges(revmer)
		rmm.extend(mismatches(revmer))
		outmsg = 'MxM,BxB,MxB'
		
	if search=='mb':
		fmm = mismatches(former)
		rmm = bulges(revmer)
		outmsg = 'MxB'
	if search=='bm':
		fmm = bulges(former)
		rmm = mismatches(revmer)
		outmsg = 'BxM'
	total = 0	
							
	p = qfunc(fmm,rmm)
	#print p
	end = time.time()
	seconds = end - begtime
	print 'Completed >> {}: {} seconds.'.format(outmsg,"%.3f" % seconds)



PerfectSearch(allchrs)


MOBSearchCUDA('d')

print '__________________'
end = time.time()
seconds = end - start
print 'Total Script Duration: {} seconds.'.format("%.3f" % seconds)


