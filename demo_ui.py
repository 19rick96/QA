import numpy as np
import theano
import theano.tensor as T
import os as os
import utils
import sys
import updates as upd
import init
import random
import cPickle as pickle
import csv
from theano.tensor.nnet import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import gensim
from gensim import corpora
import math
from nltk.corpus import stopwords
from operator import itemgetter
stop = stopwords.words('english')

import Tkinter
import tkFileDialog

srng = RandomStreams()
rng = np.random.RandomState(23455)

vocab = {}
ivocab = {}
word_vector_size = 300
word2vec = utils.load_glove(word_vector_size)
res = 1

#	GENERAL FUNCTIONS

def process_input(data1):
	q = []
	a = []
	sent = []
	for i in range(0,len(data1)):
		s1 = data1[i][0]
		s2 = data1[i][1]
		s1 = s1.strip()
		s1 = repl(s1)
		s1 = s1.lower().split(' ')
		s1 = [w for w in s1 if len(w) > 0]
		s1_vector = [utils.process_word(word = w, 
	                                word2vec = word2vec, 
	                                vocab = vocab, 
	                                ivocab = ivocab, 
	                                word_vector_size = word_vector_size, 
	                                to_return = "word2vec",silent=True) for w in s1]
		s2 = s2.strip()
		s2 = repl(s2)
		s2 = s2.lower().split(' ')
		s2 = [w for w in s2 if len(w) > 0]
		s2_vector = [utils.process_word(word = w, 
	                                word2vec = word2vec, 
	                                vocab = vocab, 
	                                ivocab = ivocab, 
	                                word_vector_size = word_vector_size, 
	                                to_return = "word2vec",silent=True) for w in s2]
		q.append(np.vstack(s1_vector))
		a.append(np.vstack(s2_vector))
		sent.append(data1[i][1])
	q = np.asarray(q)
	a = np.asarray(a)
	sent = np.asarray(sent)
	return q,a,sent

def repl(q):
	q = q.replace('.',' . ')
	q = q.replace(',',' , ')
	q = q.replace('?',' ? ')
	q = q.replace(';',' ; ')
	q = q.replace(':',' : ')
	q = q.replace('"',' " ')
	q = q.replace('(',' ( ')
	q = q.replace(')',' ) ')
	q = q.replace('$',' $ ')
	q = q.replace('!',' ! ')
	q = q.replace('^',' ^ ')
	q = q.replace("-"," - ")
	q = q.replace('}',' } ')
	q = q.replace('{',' { ')
	q = q.replace('%',' % ')
	q = q.replace("'"," ' ")
	return q

def repl_2(q):
	q = q.replace('.',' ')
	q = q.replace(',',' ')
	q = q.replace('?',' ')
	q = q.replace(';',' ')
	q = q.replace(':',' ')
	q = q.replace('"',' ')
	q = q.replace('(',' ')
	q = q.replace(')',' ')
	q = q.replace('$',' ')
	q = q.replace('!',' ')
	q = q.replace('^',' ')
	q = q.replace("-"," ")
	q = q.replace('}',' ')
	q = q.replace('{',' ')
	q = q.replace('%',' ')
	q = q.replace("'"," ")
	return q

def constant_param(value=0.0, shape=(0,)):
	return theano.shared(init.Constant(value).sample(shape), borrow=True)
   
def normal_param(std=0.1, mean=0.0, shape=(0,)):
	return theano.shared(init.Normal(std, mean).sample(shape), borrow=True)

def uniform_param(shape=(0,4),func="tanh"):
	if func == "tanh":
		P = np.random.uniform(-np.sqrt(2.0/(shape[0]+shape[1])),np.sqrt(2.0/(shape[0]+shape[1])),(shape[0],shape[1]))
	else : 
		P = np.random.uniform(-4*np.sqrt(6/(shape[0]+shape[1])),4*np.sqrt(6/(shape[0]+shape[1])),(shape[0],shape[1]))
	P = theano.shared(value=P.astype(theano.config.floatX),borrow=True)
	return P

def l2_reg(params):
    return T.sum([T.sum(x ** 2) for x in params])

def dropout(X,p=0.):
	if p>0:
		retain_prob = 1-p
		X *= srng.binomial(X.shape,p=retain_prob,dtype = theano.config.floatX)
		X /= retain_prob
	return X

def cos_sim(a,b):
	numer = sum(a*b)
	a_norm = np.sqrt(sum(a*a))
	b_norm = np.sqrt(sum(b*b))
	sim = numer/(a_norm*b_norm)
	return sim

def extract_train(s1,s2,sim):
	s1_tr = s1[0:tr_num]
	s2_tr = s2[0:tr_num]
	sim_tr = sim[0:tr_num]
	return s1_tr,s2_tr,sim_tr

def extract_test(s1,s2,sim):
	s1_test = s1[tr_num:len(s1)]
	s2_test = s2[tr_num:len(s2)]
	sim_test = sim[tr_num:len(s1)]
	return s1_test,s2_test,sim_test

def shuffle(tr_q,tr_a,tr_t):
        print "==> Shuffling the train set"
        combined = zip(tr_q,tr_a,tr_t)
        random.shuffle(combined)
        tr_q,tr_a,tr_t = zip(*combined)
	return tr_q,tr_a,tr_t

def pearson_score(p,q):
	score = (sum(p*q) - ((sum(p)*sum(q))/len(p)))/(np.sqrt((sum(p*p)-((sum(p)*sum(p))/len(p)))*(sum(q*q)-((sum(q)*sum(q))/len(q)))))
	return score

def mse(t,scr):
	return (t-scr)*(t-scr)

#	MAKE CLASSES FOR GRU, LSTM & ATTN FUNCTION LATER

def gru_next_state(x_t,s_tm1,U0,W0,b0,U1,W1,b1,U2,W2,b2):
	z_t = T.nnet.hard_sigmoid(U0.dot(x_t) + W0.dot(s_tm1) + b0)
	r_t = T.nnet.hard_sigmoid(U1.dot(x_t) + W1.dot(s_tm1) + b1)
    	h_t = T.tanh(U2.dot(x_t) + W2.dot(s_tm1 * r_t) + b2)
   	s_t = ((T.ones_like(z_t) - z_t) * h_t) + (z_t * s_tm1)
	return s_t

def lstm_next_state(x_t,h_tm1,s_tm1,Ui,Wi,bi,Uf,Wf,bf,Uo,Wo,bo,Ug,Wg,bg):
	i = T.nnet.sigmoid(Ui.dot(x_t) + Wi.dot(s_tm1) + bi)
	f = T.nnet.sigmoid(Uf.dot(x_t) + Wf.dot(s_tm1) + bf)
	o = T.nnet.sigmoid(Uo.dot(x_t) + Wo.dot(s_tm1) + bo)
	g = T.tanh(Ug.dot(x_t) + Wg.dot(s_tm1) + bg)
	h_t = (h_tm1*f) + (g*i)
	s_t = T.tanh(h_t)*o
	return h_t,s_t

#	CLASS FOR BM25

class BM25 :
    def __init__(self, fn_docs, delimiter='|') :
        self.dictionary = corpora.Dictionary()
        self.DF = {}
        self.delimiter = delimiter
        self.DocTF = []
        self.DocIDF = {}
        self.N = 0
        self.DocAvgLen = 0
        self.fn_docs = fn_docs
        self.DocLen = []
        self.buildDictionary()
        self.TFIDF_Generator()

    def buildDictionary(self) :
        raw_data = []
        for line in file(self.fn_docs) :
	    a = repl_2(line)
	    a = a.strip().lower().split(self.delimiter)
	    a = [i for i in a if i not in stop]
            raw_data.append(a)
        self.dictionary.add_documents(raw_data)

    def TFIDF_Generator(self, base=math.e) :
        docTotalLen = 0
        for line in file(self.fn_docs) :
	    doc = repl_2(line)
            doc = doc.strip().lower().split(self.delimiter)
	    doc = [i for i in doc if i not in stop]
            docTotalLen += len(doc)
            self.DocLen.append(len(doc))
            #print self.dictionary.doc2bow(doc)
            bow = dict([(term, freq*1.0/len(doc)) for term, freq in self.dictionary.doc2bow(doc)])
            for term, tf in bow.items() :
                if term not in self.DF :
                    self.DF[term] = 0
                self.DF[term] += 1
            self.DocTF.append(bow)
            self.N = self.N + 1
        for term in self.DF:
            self.DocIDF[term] = math.log((self.N - self.DF[term] +0.5) / (self.DF[term] + 0.5), base)
        self.DocAvgLen = docTotalLen / self.N

    def BM25Score(self, Query=[], k1=1.5, b=0.75) :
        query_bow = self.dictionary.doc2bow(Query)
        scores = []
        for idx, doc in enumerate(self.DocTF) :
            commonTerms = set(dict(query_bow).keys()) & set(doc.keys())
            tmp_score = []
            doc_terms_len = self.DocLen[idx]
            for term in commonTerms :
                upper = (doc[term] * (k1+1))
                below = ((doc[term]) + k1*(1 - b + b*doc_terms_len/self.DocAvgLen))
                tmp_score.append(self.DocIDF[term] * upper / below)
            scores.append(sum(tmp_score))
        return scores

    def TFIDF(self) :
        tfidf = []
        for doc in self.DocTF :
            doc_tfidf  = [(term, tf*self.DocIDF[term]) for term, tf in doc.items()]
            doc_tfidf.sort()
            tfidf.append(doc_tfidf)
        return tfidf

    def Items(self) :
        # Return a list [(term_idx, term_desc),]
        items = self.dictionary.items()
        items.sort()
        return items
 
#	CLASS FOR LEXICAL DECOMPOSITION FOLLOWED BY LSTM AND ATTN FUNCTION :

class LCD(object):
	def __init__(self,f_match,f_decomp,hidden_dim,att_hid_dim):
		fm = f_match.split("-")
		self.f_match = fm[0]
		self.fm_win = 0
		if len(fm) != 1:
			self.fm_win = int(fm[1])
		self.f_decomp = f_decomp
		self.hidden_dim = hidden_dim
		self.att_hid_dim = att_hid_dim
		
		#gru sentence parameters
		self.U0_i = normal_param(std=(2.0/(self.hidden_dim+word_vector_size)), shape=(self.hidden_dim, word_vector_size))
		self.U1_i = normal_param(std=(2.0/(self.hidden_dim+word_vector_size)), shape=(self.hidden_dim, word_vector_size))
		self.U2_i = normal_param(std=(2.0/(self.hidden_dim+word_vector_size)), shape=(self.hidden_dim, word_vector_size))
		self.W0_i = normal_param(std=(2.0/(self.hidden_dim+self.hidden_dim)), shape=(self.hidden_dim, self.hidden_dim))
		self.W1_i = normal_param(std=(2.0/(self.hidden_dim+self.hidden_dim)), shape=(self.hidden_dim, self.hidden_dim))
		self.W2_i = normal_param(std=(2.0/(self.hidden_dim+self.hidden_dim)), shape=(self.hidden_dim, self.hidden_dim))
		self.b0_i = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b1_i = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b2_i = constant_param(value=0.0, shape=(self.hidden_dim,))
		#  1 attention mechanism parameters
		self.W1 = normal_param(std=(2.0/(self.hidden_dim+(4*self.hidden_dim))), shape=(self.hidden_dim, (4*self.hidden_dim)+1))
		self.W2 = normal_param(std=(2.0/(res+self.hidden_dim)), shape=(res,self.hidden_dim))
		self.b1 = constant_param(value=0.0, shape=(self.hidden_dim,))
		self.b2 = constant_param(value=0.0, shape=(res,))
		self.Wb = normal_param(std=(2.0/(self.hidden_dim+self.hidden_dim)), shape=(self.hidden_dim,self.hidden_dim))
		#  2 attention mechanism parameters
		self.Wx = normal_param(std=(2.0/(self.att_hid_dim+(2*self.hidden_dim))), shape=(self.att_hid_dim, 2*self.hidden_dim))
		self.Wd = normal_param(std=(2.0/(self.att_hid_dim+(2*self.hidden_dim))), shape=(self.att_hid_dim, 2*self.hidden_dim))
		self.b_h = constant_param(value=0.0, shape=(self.att_hid_dim,))
		self.b_p = constant_param(value=0.0, shape=(res,))
		self.Wp = normal_param(std=(2.0/(res+self.att_hid_dim)), shape=(res,self.att_hid_dim))

		spl = T.matrix()
		smin = T.matrix()
		tpl = T.matrix()
		tmin = T.matrix()
		scr = T.scalar()
		
		s_p,spl_updates = theano.scan(self.input_next_state,sequences=spl,outputs_info=T.zeros_like(self.b0_i))
		s_m,smin_updates = theano.scan(self.input_next_state,sequences=smin,outputs_info=T.zeros_like(self.b0_i))
		t_p,tpl_updates = theano.scan(self.input_next_state,sequences=tpl,outputs_info=T.zeros_like(self.b0_i))
		t_m,tmin_updates = theano.scan(self.input_next_state,sequences=tmin,outputs_info=T.zeros_like(self.b0_i))

		s_plus = s_p[-1]
		s_minus = s_m[-1]
		t_plus = t_p[-1]
		t_minus = t_m[-1]

		s = T.concatenate([s_plus,s_minus],axis = 0)
		t = T.concatenate([t_plus,t_minus],axis=0)

		self.pred = self.attn_step_2(s,t)	

		self.loss = (scr-self.pred)*(scr-self.pred)
		#self.loss = -(scr*T.log(self.pred)) - ((1-scr)*T.log(1-self.pred))	   #for binary class for QA	

		self.params = [self.U0_i,self.W0_i,self.b0_i,
				self.U1_i,self.W1_i,self.b1_i,
				self.U2_i,self.W2_i,self.b2_i,
				self.Wx,self.Wd,self.b_h,self.b_p,self.Wp]

		#self.loss = self.loss + 0.00003*l2_reg(self.params)
		updts = upd.adam(self.loss,self.params)

		self.train_fn = theano.function(inputs = [spl,smin,tpl,tmin,scr], outputs = [self.pred,self.loss], updates = updts)
		self.test_fn = theano.function(inputs = [spl,smin,tpl,tmin], outputs = self.pred)
		#self.f = theano.function([s,t], [self.o_s,self.o_t])
	
	def input_next_state(self,x_t,s_tm1):
		s_t = gru_next_state(x_t,s_tm1,self.U0_i,self.W0_i,self.b0_i,self.U1_i,self.W1_i,self.b1_i,self.U2_i,self.W2_i,self.b2_i)
		return s_t

	"""
	def lstm_inp_next_state(self,x_t,h_tm1,s_tm1):
		h_t,s_t = lstm_next_state(x_t,h_tm1,s_tm1,self.Ui,self.Wi,self.bi,self.Uf,self.Wf,self.bf,self.Uo,self.Wo,self.bo,
						self.Ug,self.Wg,self.bg)
		return h_t,s_t	
	"""
	def attn_step(self,a,q):
		aWq = T.stack([T.dot(T.dot(a, self.Wb), q)])
		z = T.concatenate([a,q,a*q,T.abs_(a-q),aWq],axis=0)
		l_1 = T.dot(self.W1, z) + self.b1
		l_1 = T.tanh(l_1)
		l_2 = T.dot(self.W2,l_1) + self.b2
		g = T.nnet.softmax(l_2)
		return g[0]

	def attn_step_2(self,l,r):
		dot = l*r
		diff = T.abs_(l-r) 
		l_h = T.nnet.sigmoid(T.dot(self.Wx,dot) + T.dot(self.Wd,diff) + self.b_h)
		l_p = T.nnet.sigmoid(T.dot(self.Wp,l_h) + self.b_p)
		return l_p[0]

	def kl_div(self,p,q):
		res = T.sum((p*T.log(p/q)),axis=0)
		return res

	def decompose(self,s,t):
		#similarity matrix
		A = np.zeros((t.shape[0],s.shape[0]))
		for i in range(0,A.shape[0]):
			for j in range(0,A.shape[1]):
				A[i][j] = cos_sim(t[i],s[j])
		#creating semantic similarity matrix for each sentence (apply f_match) : global, local-w , max
		s_sem = []
		t_sem = []
		if self.f_match == "global":
			for i in range(0,len(s)):
				si = np.zeros(word_vector_size)
				for j in range(0,len(t)):
					si = si + (A[j][i]*t[j])
				s_sem.append(si)
			for i in range(0,len(t)):
				ti = np.zeros(word_vector_size)
				for j in range(0,len(s)):
					ti = ti + (A[i][j]*s[j])
				t_sem.append(ti)
		elif self.f_match == "local":
			At = A.T
			for i in range(0,len(s)):
				k = At[i].argmax()
				w = self.fm_win
				low = k-w
				high = k+w
				if low<0 : low = 0
				if high>len(t) : high = len(t)
				si = np.zeros(word_vector_size)
				for j in range(low,high):
					si = si + (A[j][i]*t[j])
				s_sem.append(si)
			for i in range(0,len(t)):
				k = A[i].argmax()
				w = self.fm_win
				low = k-w
				high = k+w
				if low<0 : low = 0
				if high>len(s) : high = len(s)
				ti = np.zeros(word_vector_size)
				for j in range(low,high):
					ti = ti + (A[i][j]*s[j])
				t_sem.append(ti)
		else:
			At = A.T
			for i in range(0,len(s)):
				arg = At[i].argmax()
				s_sem.append(t[arg])
			for i in range(0,len(t)):
				arg = A[i].argmax()
				t_sem.append(s[arg])
		s_sem = np.asarray(s_sem)
		t_sem = np.asarray(t_sem)
		#decomposition of similarity matrices  : rigid, linear, orthogonal
		splus = []
		sminus = []
		tplus = []
		tminus = []
		if self.f_decomp == "rigid":
			for i in range(0,len(s_sem)):
				if np.array_equal(s_sem[i],s[i]):
					splus.append(s[i])
					sminus.append(np.zeros(word_vector_size))
				else:
					sminus.append(s[i])
					splus.append(np.zeros(word_vector_size))
			for i in range(0,len(t_sem)):
				if np.array_equal(t_sem[i],t[i]):
					tplus.append(t[i])
					tminus.append(np.zeros(word_vector_size))
				else:
					tminus.append(t[i])
					tplus.append(np.zeros(word_vector_size))
		elif self.f_decomp == "linear":
			for i in range(0,len(s_sem)):
				alpha = cos_sim(s_sem[i],s[i])
				splus.append(alpha*s[i])
				sminus.append((1-alpha)*s[i])
			for i in range(0,len(t_sem)):
				alpha = cos_sim(t_sem[i],t[i])
				tplus.append(alpha*t[i])
				tminus.append((1-alpha)*t[i])
		else : 
			for i in range(0,len(s_sem)):
				coeff = sum(s_sem[i]*s[i])/sum(s_sem[i]*s_sem[i])
				splus.append(coeff*s_sem[i])
				sminus.append(s[i]-(coeff*s_sem[i]))
			for i in range(0,len(t_sem)):
				coeff = sum(t_sem[i]*t[i])/sum(t_sem[i]*t_sem[i])
				tplus.append(coeff*t_sem[i])
				tminus.append(t[i]-(coeff*t_sem[i]))
		splus = np.asarray(splus)
		sminus = np.asarray(sminus)
		tplus = np.asarray(tplus)
		tminus = np.asarray(tminus)
		return splus,sminus,tplus,tminus

	def l1(self,s1w,s2w):
		return -T.abs_(s1w-s2w)

	def save_params(self, file_name, epoch):
		with open(file_name, 'w') as save_file:
		    pickle.dump(
		        obj = {
		            'params' : [x.get_value() for x in self.params],
		            'epoch' : epoch, 
		        },
		        file = save_file,
		        protocol = -1
		    )
    
	def load_state(self, file_name):
		print "==> loading state %s" % file_name
		with open(file_name, 'r') as load_file:
			dict = pickle.load(load_file)
			loaded_params = dict['params']
			for (x, y) in zip(self.params, loaded_params):
				x.set_value(y)

	def train(self,s1,s2,sim_scr,itr):
		l = len(s1)
		print "starting..."
		for j in range(0,itr):
			a_loss = 0.0
			#tr_q,tr_a,tr_t = shuffle(tr_q,tr_a,tr_t)
			for i in range(0,l):
				sp,sm,tp,tm = self.decompose(s1[i],s2[i])
				pred,loss = self.train_fn(sp,sm,tp,tm,sim_scr[i])
				a_loss=a_loss+loss
				print "iteration : %d , %d" %((i+1),(j+1))
				print "loss : %.3f  average_loss : %.3f "%(loss,a_loss/(i+1))
				print "******************"
				if ((i+1)%10 == 0):
					fname = 'final/lex_rnn_attn_eqa.epoch%d.txt' %(j)
					self.save_params(fname,j)


#	PERFORM RANKING : BM25 AND THEN LCD

lcd = LCD("max","orthogonal",150,80)
lcd.load_state("final/lex_rnn_attn.epoch3")

class SampleApp(Tkinter.Tk):
    def __init__(self):
        Tkinter.Tk.__init__(self)
	self.minsize(width=1500, height=900)
	self.button_browsefile = Tkinter.Button(self, text='Select file...',command=self.open_file_dialog).pack()
	self.filename = "a.txt"
	self.query = "how are u ?"
	self.dis_file = Tkinter.Text(self,height = 1)
	self.dis_file.insert(Tkinter.INSERT,self.filename)
	self.dis_file.pack()
	self.entry = Tkinter.Entry(self,width = 100)
        self.button_q = Tkinter.Button(self, text="Enter question and Press ", command=self.on_button)
        self.button_q.pack()
        self.entry.pack()
	self.dis_output = Tkinter.Text(self,height = 50,width = 200)
	self.dis_output.pack()

    def on_button(self):
	self.query = self.entry.get()
	self.dis_output.delete('1.0',Tkinter.END)
	self.dis_output.insert(Tkinter.INSERT,"calculating ... \n")
	bm25 = BM25(self.filename, delimiter=' ')
	para = []
	for line in file(self.filename):
		para.append(line)
	Query_org = self.query
	Query = repl(Query_org)
	Query = Query.strip().lower().split()
	Query = [i for i in Query if i not in stop]
	scores = bm25.BM25Score(Query)
	tfidf = bm25.TFIDF()

	nn_candidate = []
	for i in range(0,len(scores)):
		if scores[i] >= 0.1:
			row = []
			row.append(Query_org)
			row.append(para[i])
			row.append(scores[i])
			nn_candidate.append(row)

	nn_candidate = sorted(nn_candidate,key=itemgetter(0),reverse=True)
	n = 15
	if n>len(nn_candidate):
		n = len(nn_candidate)
	self.dis_output.insert(Tkinter.INSERT,"\nRANKING AFTER BM25 : ")
	self.dis_output.insert(Tkinter.INSERT,"\n ############################################################################### ")
	for i in range(0,n):
		self.dis_output.insert(Tkinter.INSERT,"\n" + nn_candidate[i][1])
	self.dis_output.insert(Tkinter.INSERT,"\n ############################################################################### ")
	self.dis_output.pack()

	q,a,sentences = process_input(nn_candidate)
	self.dis_output.insert(Tkinter.INSERT,"\n calculating ... ")
	final_ranking = []
	for i in range(0,len(a)):
		sp,sm,tp,tm = lcd.decompose(q[i],a[i])
		pr = lcd.test_fn(sp,sm,tp,tm)
		row = []
		row.append(pr)
		row.append(sentences[i])
		final_ranking.append(row)

	final_ranking = sorted(final_ranking,key=itemgetter(0),reverse=True)
	n = 15
	if n>len(final_ranking):
		n = len(final_ranking)
	self.dis_output.insert(Tkinter.INSERT,"\n")
	self.dis_output.insert(Tkinter.INSERT,"\nRANKING AFTER LEXICAL COMPOSITION DECOMPOSITION : ")
	self.dis_output.insert(Tkinter.INSERT,"\n ############################################################################### ")
	for i in range(0,n):
		self.dis_output.insert(Tkinter.INSERT,"\n")
		self.dis_output.insert(Tkinter.INSERT,final_ranking[i][1])
		self.dis_output.insert(Tkinter.INSERT,"\n")
		self.dis_output.insert(Tkinter.INSERT,final_ranking[i][0])
	self.dis_output.insert(Tkinter.INSERT,"\n ############################################################################### ")
        self.dis_output.pack()

    def open_file_dialog(self):
	fname = tkFileDialog.askopenfilename(filetypes= [("allfiles","*")])
	if fname == "":
		fname = 'a.txt'
	self.dis_file.delete('1.0',Tkinter.END)
	self.filename = fname
	self.dis_file.insert(Tkinter.INSERT,self.filename)
	self.dis_file.pack()

app = SampleApp()
app.mainloop()
